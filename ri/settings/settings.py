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

PROJECT_DIR: str = os.getenv("PROJECTDIR", "/tmp")

PATCH_LOGITS_CACHE_DIR: str = os.path.join(PROJECT_DIR, "patch_logits")
