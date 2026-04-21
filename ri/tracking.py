"""Opt-in W&B experiment tracking.

All methods are no-ops when tracking is disabled (``enabled=False`` or
``WANDB_MODE=disabled``). Local JSON output is always written independently
by the runners — this module is supplementary.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Thin W&B wrapper for experiment tracking."""

    def __init__(
        self,
        project: str = "when-cot-fails",
        name: str | None = None,
        tags: list[str] | None = None,
        enabled: bool | None = None,
        **kwargs: Any,
    ) -> None:
        if enabled is None:
            enabled = os.environ.get("WANDB_MODE", "").lower() != "disabled"

        self._enabled = enabled
        self._run = None

        if self._enabled:
            try:
                import wandb  # noqa: PLC0415 — soft dependency; init lazily so disabled runs never import it

                self._wandb = wandb
                self._run = wandb.init(
                    project=project,
                    name=name,
                    tags=tags or [],
                    reinit=True,
                )
            except Exception:
                logger.warning("Failed to initialize W&B — tracking disabled", exc_info=True)
                self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._run is not None

    def log_config(self, config: BaseModel) -> None:
        """Log a Pydantic model as ``wandb.config``."""
        if not self.is_enabled:
            return
        self._run.config.update(config.model_dump(), allow_val_change=True)  # type: ignore[union-attr]

    def log_metrics(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        """Log scalar metrics via ``wandb.log()``."""
        if not self.is_enabled:
            return
        self._wandb.log(metrics, step=step)

    def log_table(self, table_name: str, rows: Sequence[BaseModel]) -> None:
        """Log a list of Pydantic models as a ``wandb.Table``."""
        if not self.is_enabled or not rows:
            return
        columns = list(rows[0].model_fields.keys())
        table = self._wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[_serialize_value(v) for v in row.model_dump().values()])
        self._wandb.log({table_name: table})

    def log_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Log a pandas DataFrame as a ``wandb.Table``.

        Lists/arrays inside cells are stringified — the usual wandb Table
        footgun that silently drops nested structures otherwise.
        """
        if not self.is_enabled or df.empty:
            return
        columns = list(df.columns)
        table = self._wandb.Table(columns=columns)
        for row in df.itertuples(index=False, name=None):
            table.add_data(*[_serialize_value(v) for v in row])
        self._wandb.log({table_name: table})

    def log_figure(self, key: str, fig: Figure) -> None:
        """Log a matplotlib figure as ``wandb.Image``."""
        if not self.is_enabled:
            return
        self._wandb.log({key: self._wandb.Image(fig)})

    def finish(self) -> None:
        if self.is_enabled:
            self._wandb.finish()
            self._run = None


def _serialize_value(value: Any) -> Any:
    """Convert a value to a W&B-compatible type.

    wandb Tables silently drop or corrupt nested lists/dicts; stringifying
    them is the simplest reliable fix.
    """
    if isinstance(value, (list, tuple, dict)):
        return str(value)
    return value
