from typing import Any


def prompt_text_from_rendered(rendered_prompt: Any) -> str:
    """Extract the text content from a rendered prompt."""
    if isinstance(rendered_prompt, str):
        return rendered_prompt
    if isinstance(rendered_prompt, (list, tuple)):
        for message in reversed(rendered_prompt):
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
    return ""
