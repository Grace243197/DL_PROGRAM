# W10 model environment and skeleton

This branch adds the PyTorch side that W9 intentionally left out.  It does not
read the UTA-RLDD window files yet, so the model tests can run before the
19.69 GiB handoff is available.

## Locked environment

Validated on the project machine:

| Item | Value |
|---|---|
| Python | 3.10.22 |
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB |
| Driver | 621.14 |
| PyTorch | 2.14.1+cu126 |
| torchvision | 0.29.1+cu126 |
| NumPy | 1.26.4 (`<2` keeps MediaPipe 0.10.21 compatible) |

On the project machine the pinned dependencies live in the dedicated conda
environment `dl_env` (Python 3.10.22), isolated from the shared `env1`:

```powershell
conda activate dl_env
python -m pip install -r requirements.txt
```

`requirements.txt` includes both the W9 data dependencies and the model
dependencies.  The CUDA wheel index is pinned in `requirements-model.txt`;
do not replace `+cu126` with a plain `torch` install, or pip may resolve a
CPU-only wheel.  To rebuild the interpreter from scratch,
`conda create -n dl_env python=3.10 pip`, then install `requirements.txt`;
package versions come from that file, not from conda.

Verify the GPU:

```powershell
python scripts/check_model_env.py
```

## Model cells

`src/model/` implements the four cells without coupling them to the dataset:

```python
from src.model import build_backbone, build_model

backbone = build_backbone(config)                    # frozen MobileNetV3-Small, 576-d
frame     = build_model(config, temporal=False, use_prior=False)  # CNN-Frame
frame_film= build_model(config, temporal=False, use_prior=True)   # CNN-Frame-Prior
gru       = build_model(config, temporal=True,  use_prior=False)  # CNN-GRU
gru_film  = build_model(config, temporal=True,  use_prior=True)   # CNN-GRU-Fusion
```

Each head accepts `frame_features [B, T, 576]`, `valid_mask [B, T]`, and
`priors [B, 34]`; it returns `ModelOutput(logits, gate)`.  `gate` is the FiLM
modulation vector used by the mechanism probe.  `fusion="concat"` is the
ablation path.

## BGR -> RGB contract

OpenCV decodes video as BGR, and the saved 160x160 crops keep that order.
`src/data/image_ops.py::bgr_to_rgb` swaps the last axis and returns a
C-contiguous copy.  The synthetic-color test catches the common mistake of
feeding BGR crops into an ImageNet-pretrained RGB backbone.

## Tests

```powershell
python -m pytest tests/test_bgr_rgb.py tests/test_model_skeleton.py -q
```

The model tests use `pretrained=False`, so they do not download ImageNet
weights.  The real training entry should use `pretrained=True` from
`configs/default.yaml`.
