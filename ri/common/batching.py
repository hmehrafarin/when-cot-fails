from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ri.utils.tokenizer import build_role_header

USER_MARKER = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
ASSISTANT_MARKER = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

#!TODO: refactor to drop the token importance, we will remove it in the next update


def _normalize_qa_item(
    item: Dict[str, Any],
    *,
    user_marker: str,
    assistant_marker: str,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Normalize a dataset item into (question, answer, token_importance_list).
    """
    question = item.get("question", "")
    answer = item.get("answer", "")

    ti_raw = item.get("token_importance", []) or []
    start_idx = 0
    end_idx = len(ti_raw)
    for i, t in enumerate(ti_raw):
        if isinstance(t, dict) and t.get("token") == user_marker:
            start_idx = i + 1
            break
    for j in range(start_idx, len(ti_raw)):
        t = ti_raw[j]
        if isinstance(t, dict) and t.get("token") == assistant_marker:
            end_idx = j
            break
    segment = ti_raw[start_idx:end_idx]
    cleaned_ti: List[Dict[str, Any]] = []
    for t in segment:
        if not isinstance(t, dict):
            continue
        tok = str(t.get("token", ""))
        try:
            pos = int(t.get("pos", 0))
        except Exception:
            pos = 0
        try:
            score = float(t.get("score", 0.0))
        except Exception:
            score = 0.0
        cleaned_ti.append({"token": tok, "pos": pos, "score": score})
    return question, answer, cleaned_ti


def prepare_batch_data(
    data,
    batch_idx: int,
    batch_size: int,
    include_importance: bool = True,
    tokenizer: Any | None = None,
):
    """
    Slice the data for the given batch index and return questions, answers, and
    a list of dictionaries for prompts.
    """
    start = max(0, batch_idx * batch_size)
    n = len(data)
    if start >= n:
        return [], [], []

    end = min(start + batch_size, n)

    batched_input = []
    batch_questions: List[str] = []
    batch_answers: List[str] = []

    user_marker = build_role_header(tokenizer, "user") or USER_MARKER
    assistant_marker = build_role_header(
        tokenizer, "assistant") or ASSISTANT_MARKER

    for i in range(start, end):
        item = data[i] or {}
        q, a, ti = _normalize_qa_item(
            item,
            user_marker=user_marker,
            assistant_marker=assistant_marker,
        )

        entry = {"question": q, "answer": a}
        if include_importance:
            entry["token_importance"] = ti

        batched_input.append(entry)
        batch_questions.append(q)
        batch_answers.append(a)

    return batch_questions, batch_answers, batched_input
