# SPDX-License-Identifier: Apache-2.0
from functools import cache

import torch


@cache
def is_gfx936(device: int | torch.device) -> bool:
    """Return whether *device* is the validated gfx936 target."""
    return torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx936:")


def use_gfx936(tensor: torch.Tensor) -> bool:
    """Cheap hot-path target check shared by every framework adapter."""
    return tensor.is_cuda and is_gfx936(tensor.device)
