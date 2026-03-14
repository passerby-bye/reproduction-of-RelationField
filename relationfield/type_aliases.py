"""Local type aliases with runtime-safe fallbacks."""

import torch

try:
    from torchtyping import TensorType as _TensorType

    TensorType = _TensorType
except Exception:
    class TensorType:
        """Fallback that keeps TensorType[...] annotations import-safe."""

        def __class_getitem__(cls, _item):
            return torch.Tensor
