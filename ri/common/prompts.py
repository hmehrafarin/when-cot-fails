from __future__ import annotations

from typing import Literal

from ri.prompts.prompter import Prompter

StepsLiteral = int | Literal["all"] | Literal["no_steps"]


def build_prompt_batch(
    prompter: Prompter,
    batch_row: list[dict[str, str | None]],
    steps: StepsLiteral | None = None,
):
    """
    Build a batch of prompts based on the prompter template.
    """

    def _slice_answer(ans: str | None) -> str:
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

    adjusted_rows: list[dict[str, str | None]] = []
    for row in batch_row:
        q = row.get("question", "")
        a = row.get("answer", None)
        adjusted_rows.append(
            {
                "question": q,
                "answer": _slice_answer(a),
            }
        )

    system_text = prompter.template.get(
        "system",
        "you are a step-by-step reasoner, finish the reasoning with\nFinal Answer: #### [number]",
    )
    user_prompts = prompter.create_prompt(adjusted_rows)

    return [
        [
            {"role": "system", "content": system_text},
            {"role": "user", "content": up},
        ]
        for up in user_prompts
    ]
