from __future__ import annotations

from typing import Any


def prepare_batch_data(
    data: Any,
    batch_idx: int,
    batch_size: int,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Slice ``data`` for ``batch_idx`` and return ``(questions, answers, prompt_rows)``.

    Each prompt row is a ``{"question": ..., "answer": ...}`` dict consumed by
    :func:`ri.common.prompts.build_prompt_batch`.
    """
    start = max(0, batch_idx * batch_size)
    n = len(data)
    if start >= n:
        return [], [], []

    end = min(start + batch_size, n)

    batch_questions: list[str] = []
    batch_answers: list[str] = []
    batched_input: list[dict[str, Any]] = []

    for i in range(start, end):
        item = data[i] or {}
        question = item.get("question", "")
        answer = item.get("answer", "")

        batched_input.append({"question": question, "answer": answer})
        batch_questions.append(question)
        batch_answers.append(answer)

    return batch_questions, batch_answers, batched_input
