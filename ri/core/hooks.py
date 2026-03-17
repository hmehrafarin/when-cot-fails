from typing import Any, Dict, List, Optional, Union

import torch

from .model import ModelAndTokenizer


def set_patch(
    model: ModelAndTokenizer,
    patch_config: List[Dict[str, Any]],
) -> List[Any]:
    """Register forward hooks on model layers to replace hidden states at given positions."""

    def _make_hook(
        hs_position: List[Union[Dict[str, int], int]],
        hs: Optional[List[torch.Tensor]] = None,
    ):

        def _unpack(output):
            if isinstance(output, tuple):
                lst = list(output)
                x = lst[0]

                def repack(nx):
                    lst[0] = nx
                    return tuple(lst)

                return x, repack
            if hasattr(output, "last_hidden_state"):
                x = output.last_hidden_state

                def repack(nx):
                    output.last_hidden_state = nx
                    return output

                return x, repack
            x = output

            def repack(nx):
                return nx

            return x, repack

        def hook_fn(module, inputs, output):
            x, repack = _unpack(output)
            if not isinstance(x, torch.Tensor) or x.dim() != 3:
                return output

            B, T, H = x.shape

            if T <= 1:
                return output

            assert hs is not None, "'hs' must be provided for 'replace'"
            y = x.clone()

            for i in range(B):
                pos_i = hs_position[i]
                if isinstance(pos_i, int):
                    pos_list = [pos_i]
                else:
                    pos_list = list(pos_i)
                pos_list = [p + T if p < 0 else p for p in pos_list]
                pos_list = [max(0, min(p, T - 1)) for p in pos_list]

                vec_i = hs[i]
                if isinstance(vec_i, torch.Tensor):
                    vec_tensor = vec_i.to(y.device, dtype=y.dtype)
                    if vec_tensor.dim() == 1:
                        vec_list = [vec_tensor]
                    elif vec_tensor.dim() == 2:
                        mcount = vec_tensor.shape[0]
                        assert mcount == len(pos_list), (
                            f"'hs' (tensor) count {mcount} must equal number of positions "
                            f"{len(pos_list)} for sample {i}"
                        )
                        vec_list = [vec_tensor[m] for m in range(mcount)]
                    else:
                        continue
                elif isinstance(vec_i, (list, tuple)):
                    vec_list = []
                    for v in vec_i:
                        if isinstance(v, torch.Tensor):
                            vec_list.append(v.to(y.device, dtype=y.dtype))
                        else:
                            vec_list.append(
                                torch.tensor(
                                    v, device=y.device, dtype=y.dtype)
                            )
                else:
                    vec_list = [
                        torch.tensor(vec_i, device=y.device, dtype=y.dtype)
                    ]

                for j, pos in enumerate(pos_list):
                    if j >= len(vec_list):
                        raise AssertionError(
                            f"Not enough hidden states to unpack. "
                            f"Vector length:{len(vec_list)} < position length:{len(pos_list)}"
                        )
                    y[i, pos, :] = vec_list[j]
            return repack(y)

        return hook_fn

    hooks = []
    for cfg in patch_config:
        hook = _make_hook(
            hs_position=cfg["hs_position"],
            hs=cfg.get("hs", None),
        )
        layers = cfg["layer_to_patch"]
        if isinstance(layers, int):
            layers = [layers]
        for layer_idx in layers:
            hooks.append(
                model.model.layers[layer_idx].register_forward_hook(hook))
    return hooks


def remove_hooks(hooks: List[Any]) -> None:
    for hook in hooks:
        hook.remove()
