from __future__ import annotations

from ri.prompts.prompter import Prompter


def build_prompt_batch(
    prompter: Prompter,
    batch_row: list[dict[str, str | None]],
) -> list[list[dict[str, str]]]:
    """Build a batch of system/user chat turns from raw question/answer rows."""

    adjusted_rows: list[dict[str, str | None]] = [
        {"question": row.get("question", ""), "answer": ""} for row in batch_row
    ]

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
