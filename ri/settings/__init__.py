# For backwards compatibility, expose settings as a module-level object
from . import settings
from .settings import (
    DEFAULT_MODEL_NAME,
    HUGGINGFACE_CACHE_DIR,
    MODEL_CACHE_DIR,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "HUGGINGFACE_CACHE_DIR",
    "MODEL_CACHE_DIR",
    "settings",
]
