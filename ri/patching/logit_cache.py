from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class LogitCacheConfig:
    """Configuration for logit caching."""

    cache_dir: str
    sample_idx: int
    source_model_name: str
    target_model_name: str
    source_dataset: str
    target_dataset: str
    max_gen_len: int
    seed: int
    src_prompt_template: str
    tgt_prompt_template: str

    def compute_cache_hash(self) -> str:
        """Compute MD5 hash from cache key components."""
        key_dict = {
            "sample_idx": self.sample_idx,
            "source_model": self.source_model_name,
            "target_model": self.target_model_name,
            "source_dataset": os.path.basename(self.source_dataset),
            "target_dataset": os.path.basename(self.target_dataset),
            "max_gen_len": self.max_gen_len,
            "seed": self.seed,
            "src_template": self.src_prompt_template,
            "tgt_template": self.tgt_prompt_template,
        }
        key_str = json.dumps(key_dict, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_cache_path(self) -> str:
        """Get the cache directory path for this configuration."""
        cache_hash = self.compute_cache_hash()
        return os.path.join(self.cache_dir, cache_hash)


class LogitCache:
    """Manages caching of patch sweep logits."""

    def __init__(self, config: LogitCacheConfig):
        self.config = config
        self.cache_path = config.get_cache_path()
        self._metadata_written = False

    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        os.makedirs(self.cache_path, exist_ok=True)

        # Write metadata on first access
        if not self._metadata_written:
            metadata_path = os.path.join(self.cache_path, "metadata.json")
            if not os.path.exists(metadata_path):
                metadata = {
                    "sample_idx": self.config.sample_idx,
                    "source_model": self.config.source_model_name,
                    "target_model": self.config.target_model_name,
                    "source_dataset": self.config.source_dataset,
                    "target_dataset": self.config.target_dataset,
                    "max_gen_len": self.config.max_gen_len,
                    "seed": self.config.seed,
                    "src_template": self.config.src_prompt_template,
                    "tgt_template": self.config.tgt_prompt_template,
                    "cache_hash": self.config.compute_cache_hash(),
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            self._metadata_written = True

    def _get_src_pos_path(self, src_pos: int) -> str:
        """Get file path for a source position's cached logits."""
        return os.path.join(self.cache_path, f"src_pos_{src_pos}.pt")

    def has_cached_logits(self, src_pos: int) -> bool:
        """Check if logits are cached for a source position."""
        return os.path.exists(self._get_src_pos_path(src_pos))

    def load_logits(
        self,
        src_pos: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Load cached logits for a source position.

        Returns:
            Tensor of shape (num_layers, num_tgt_pos, vocab_size) in float16,
            or None if not cached.
        """
        cache_file = self._get_src_pos_path(src_pos)
        if not os.path.exists(cache_file):
            return None

        try:
            cached_data = torch.load(cache_file, map_location=device, weights_only=True)
            return cached_data["logits"]
        except Exception as e:
            print(f"Warning: Failed to load cache {cache_file}: {e}")
            return None

    def save_logits(
        self,
        src_pos: int,
        logits: torch.Tensor,
        num_layers: int,
        target_positions: List[int],
    ) -> None:
        """
        Save logits for a source position.

        Args:
            src_pos: Source position index
            logits: Tensor of shape (num_layers, num_tgt_pos, vocab_size)
            num_layers: Number of layers
            target_positions: List of target position indices
        """
        self._ensure_cache_dir()
        cache_file = self._get_src_pos_path(src_pos)

        # Convert to float16 for storage efficiency
        logits_fp16 = logits.half().cpu()

        torch.save(
            {
                "logits": logits_fp16,
                "num_layers": num_layers,
                "target_positions": target_positions,
                "src_pos": src_pos,
            },
            cache_file,
        )

    def clear_cache(self) -> None:
        """Remove all cached files for this configuration."""
        import shutil

        if os.path.exists(self.cache_path):
            shutil.rmtree(self.cache_path)

    def get_cache_info(self) -> dict:
        """Get information about the cache state."""
        if not os.path.exists(self.cache_path):
            return {"exists": False, "num_files": 0, "cache_path": self.cache_path}

        pt_files = [f for f in os.listdir(self.cache_path) if f.endswith(".pt")]
        return {
            "exists": True,
            "num_files": len(pt_files),
            "cache_path": self.cache_path,
            "cache_hash": self.config.compute_cache_hash(),
        }
