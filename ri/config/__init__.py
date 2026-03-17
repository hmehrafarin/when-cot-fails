from .settings import (
    HUGGINGFACE_CACHE_DIR,
    MODEL_CACHE_DIR,
    MODEL_OUTPUT_DIR,
    DEFAULT_MODEL_NAME,
)

# For backwards compatibility, expose settings as a module-level object
from . import settings

__all__ = [
    "settings",
    "HUGGINGFACE_CACHE_DIR",
    "MODEL_CACHE_DIR",
    "MODEL_OUTPUT_DIR",
    "DEFAULT_MODEL_NAME",
]
