import os

# --------------------------------------------------------------------------- #
# Cache directories
# --------------------------------------------------------------------------- #

HUGGINGFACE_CACHE_DIR: str = (
    os.getenv("RI_CACHE_DIR")
    or os.getenv("TRANSFORMERS_CACHE")
    or os.getenv("HF_HOME")
    or os.getenv("HUGGINGFACE_HUB_CACHE")
    or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
)

# Alias for model cache (same as HuggingFace cache)
MODEL_CACHE_DIR: str = HUGGINGFACE_CACHE_DIR

MODEL_OUTPUT_DIR: str = os.getenv("RI_OUTPUT_DIR", "outputs")

SCRATCH_DIR: str = os.getenv("SCRATCHDIR", "/tmp")
PROJECT_DIR: str = os.getenv("PROJECTDIR", "/tmp")

PATCH_LOGITS_CACHE_DIR: str = os.path.join(PROJECT_DIR, "patch_logits")

# --------------------------------------------------------------------------- #
# Default model configuration
# --------------------------------------------------------------------------- #

DEFAULT_MODEL_NAME: str = os.getenv(
    "RI_DEFAULT_MODEL",
    "meta-llama/Llama-3.1-8B-Instruct",
)

# --------------------------------------------------------------------------- #
# Experiment defaults
# --------------------------------------------------------------------------- #

DEFAULT_SEED: int = 42
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_MAX_GEN_LEN: int = 400


# --------------------------------------------------------------------------- #
# Legacy compatibility: Constants class
# --------------------------------------------------------------------------- #


class Constants:
    """
    Legacy configuration class for backwards compatibility.

    Prefer using module-level constants directly instead.
    """

    HUGGINGFACE_CACHE_DIR = HUGGINGFACE_CACHE_DIR
    MODEL_CACHE_DIR = MODEL_CACHE_DIR
    MODEL_OUTPUT_DIR = MODEL_OUTPUT_DIR
