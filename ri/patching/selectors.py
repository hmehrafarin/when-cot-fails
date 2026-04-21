from __future__ import annotations

from collections.abc import Iterable


def select_position_with_mode(
    available_positions: Iterable[int],
    mode: int,
) -> int | None:
    """Pick a single position from the pool at the given integer index.

    Negative indices count from the end. Out-of-range indices clamp to the
    nearest edge. Returns None when the pool is empty.
    """
    unique: list[int] = []
    seen: set[int] = set()
    for pos in available_positions:
        try:
            ipos = int(pos)
        except (TypeError, ValueError):
            continue
        if ipos in seen:
            continue
        unique.append(ipos)
        seen.add(ipos)

    if not unique:
        return None

    idx = mode
    if idx < 0:
        idx += len(unique)
    idx = max(0, min(idx, len(unique) - 1))
    return unique[idx]
