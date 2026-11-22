"""Fail fast if the W10 PyTorch/CUDA environment is not the pinned one."""

from __future__ import annotations

import sys

import torch
import torchvision


def main() -> int:
    print(f"python      {sys.version.split()[0]}")
    print(f"torch       {torch.__version__}")
    print(f"torchvision {torchvision.__version__}")
    print(f"cuda wheel  {torch.version.cuda}")
    print(f"cudnn       {torch.backends.cudnn.version()}")
    if not torch.cuda.is_available():
        print("CUDA is not available to PyTorch", file=sys.stderr)
        return 1
    print(f"gpu         {torch.cuda.get_device_name(0)}")
    print(f"capability  {torch.cuda.get_device_capability(0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
