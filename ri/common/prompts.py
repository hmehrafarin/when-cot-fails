from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from ri.prompts.prompter import Prompter


StepsLiteral = Union[int, Literal["all"], Literal["no_steps"]]


def build_prompt_batch(
    prompter: Prompter,
    batch_row: List[Dict[str, Optional[str]]],
    steps: Optional[StepsLiteral] = None,
):
    """
    Build a batch of prompts based on the prompter template.
    """

    def _slice_answer(ans: Optional[str]) -> str:
        if not ans or steps in (None, "no_steps"):
            return ""
        lines = [ln.strip() for ln in ans.split("\n") if ln.strip()]
        if steps == "all":
            return "\n".join(lines)
        if isinstance(steps, int):
            if steps <= 0:
                return ""
            return "\n".join(lines[:steps])
        raise ValueError(f"Unsupported steps value: {steps!r}")

    adjusted_rows: List[Dict[str, Optional[str]]] = []
    for row in batch_row:
        q = row.get("question", "")
        a = row.get("answer", None)
        adjusted_rows.append({
            "question": q,
            "answer": _slice_answer(a),
        })

    system_text = prompter.template.get(
        "system",
        "you are a step-by-step reasoner, finish the reasoning with\nFinal Answer: #### [number]",
    )
    user_prompts = prompter.create_prompt(adjusted_rows)

    convos = []
    for up in user_prompts:
        convos.append([
            {"role": "system", "content": system_text},
            {"role": "user", "content": up},
        ])
    return convos
