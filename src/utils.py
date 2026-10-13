"""
utils.py: small helpers used by every other file.

Where this fits in the pipeline:
    data -> model -> train -> evaluate -> predict
None of those steps live here. This file holds the shared tools they all use:
fixing random seeds and choosing which hardware (CPU or GPU) to run on.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Make every random choice repeatable.

    Python, NumPy and PyTorch each have their OWN random number generator,
    so we have to seed all three. After this, running the same code twice
    gives the same data split, the same starting weights and the same results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Seed every GPU too (does nothing if there is no GPU).
    torch.cuda.manual_seed_all(seed)

    # cuDNN is NVIDIA's library of fast GPU maths. By default it picks the
    # fastest algorithm, which can differ slightly from run to run. These two
    # lines trade a little speed for repeatable results.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Pick the fastest hardware available to run the model on.

    A "device" is where the maths happens:
      - "cuda": an NVIDIA GPU (what Kaggle gives us). Much faster.
      - "mps":  the GPU inside Apple Silicon Macs.
      - "cpu":  the normal processor. Always available, slowest.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
