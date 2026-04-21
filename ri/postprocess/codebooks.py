from __future__ import annotations

from .generation_labels import generation_type_codes
from .spacy_rules import PRIORITY

ENTITY_ROLE_CODES: dict[str, str] = {
    "OTHER": "Fallback label when no stronger rule applies.",
    "PUNCT": "Punctuation or separator token outside arithmetic symbols.",
    "VERB": "Verb or auxiliary token in the reasoning trace.",
    "ENTITY": "Entity mention seeded from the question context or pronoun rules.",
    "NUMBER": "Numeric token that is not promoted to a quantity, operand, or result.",
    "QUANTITY": "Numeric modifier paired with a unit-bearing noun phrase.",
    "UNIT": "Measurement, currency, or rate unit token.",
    "STEP": "Explicit step marker such as `Step 2` or a numbered line prefix.",
    "OPERAND": "Operand token adjacent to an arithmetic operator or equation.",
    "RESULT": "Result token on the right-hand side of an equation.",
    "OPERATOR": "Arithmetic operator token such as `+`, `-`, `*`, `/`, or `x`.",
    "EQUALS": "Equality symbol token `=`.",
}

ENTITY_ROLE_PRIORITY: dict[str, int] = PRIORITY

GENERATION_TYPE_CODES: dict[str, str] = generation_type_codes()
