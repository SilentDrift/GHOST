from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python and NumPy and, if available, PyTorch for reproducibility.

    The function imports :mod:`torch` lazily so that it can be used even when
    PyTorch is not installed. Torch-specific seeding is only performed when the
    import succeeds.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:  # pragma: no cover - optional dependency
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    os.environ["PYTHONHASHSEED"] = str(seed)
