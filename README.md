# Anchor-Conditioned Diffusion for Scalable 3D Microstructure Synthesis

Three-dimensional imaging is expensive, so material structures are often known only through a few measured 2D sections. Models trained only to match 2D section distributions do not ensure that a supplied section influences a prescribed 3D location. Pasting it into the result afterward can create an artificial discontinuity.

This project supplies measured sections as learned conditions throughout denoising, encouraging phase agreement and natural continuity without overwriting generated values. The conditioned volume can then guide jointly generated overlapping blocks. During scale-up, the base is softly conditioned at every reverse step and remains free to adapt to its new surroundings; it is not pasted back into the final volume.

[`PAPER.md`](PAPER.md) records the evaluated method and its limitations.

## Installation

```bash
git clone https://github.com/phykn/diffusion-gan3d.git
cd diffusion-gan3d
pip install -r requirements.txt
```

## Usage

Set the 2D section folders and training options in [`config/train.yaml`](config/train.yaml), then train the model:

Each numeric domain contains its own folders for axes 0, 1, and 2:

~~~yaml
data:
  domains:
    0:
      0: [data/domain_0/axis_0]
      1: [data/domain_0/axis_1]
      2: [data/domain_0/axis_2]
    1:
      0: [data/domain_1/axis_0]
      1: [data/domain_1/axis_1]
      2: [data/domain_1/axis_2]
~~~

Folders listed under one axis are pooled within that domain. Training uniformly
samples one domain per step and takes all three axis batches from it.

Set `data.allow_partial_crop: true` to train from a section whose height or
width is smaller than `crop_size`. Each available dimension is cropped to at
most `crop_size` and resized with the common `input_size / crop_size` scale, so
the aspect ratio and physical voxel scale are preserved. The critic compares a
generated section window with the same rectangular shape. Omitting the option,
or setting it to `false`, keeps the strict error for undersized images. Images
pooled under one domain and axis must produce the same shape for batching.
Physical resolution must remain consistent across the dataset.

```bash
python run_train.py --device cuda
# Or choose the exact output directory (it must not already exist):
python run_train.py --device cuda --run-dir run/my-experiment
```

Generate a 3D volume around a known section and extend it to `3 × 3 × 3` blocks:

```python
from src.anchor import PlaneAnchor
from src.build import load_generator
from src.scale import ScaledGenerator

generator = load_generator(generator_path, device)

anchor = PlaneAnchor(image, axis=0, index=32)
base = generator.generate(anchors=(anchor,), domain=0)

volume = ScaledGenerator(generator).generate(
    blocks=(3, 3, 3),
    overlap=8,
    base=base,
    domain=0,
)
```

Single-domain weights default to domain 0; multi-domain weights should specify a domain. Direct generation produces `input_size³`. Scale-up length per axis is `input_size + (blocks - 1) × (input_size - 2 × overlap)`, so 128-sized blocks with three blocks and overlap eight produce `352³`.

`anchor_strength=0` disables anchor conditioning. `guidance_scale=1` is the standard conditional path; validate non-default guidance with weights trained using condition dropout. See [`PAPER.md`](PAPER.md) for details.


Run the diagnostic scripts with an explicit generator weight. Script `03`
always requires a GT volume; script `04` requires one when `--count` is positive
and `--anchor-strength` is nonzero:

```bash
python scripts/01_check_dataset.py
python scripts/02_check_generated.py --weight run/<run-id>/generator.pt --domain 0
python scripts/03_check_anchor.py --weight run/<run-id>/generator.pt --domain 0 --gt scripts/gt_128.tiff --count 3
python scripts/04_check_scale_up.py --weight run/<run-id>/generator.pt --domain 0
```

Generator scripts default to `--domain 0` and `--guidance-scale 1.0`. Paper scripts require explicit weights and record input provenance.

`generator.pt` is the latest weight; `checkpoint_every_steps` preserves numbered sets under `checkpoints/step_XXXXXXXX/`.

## Development

Install the development dependencies and run the regression suite:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check src scripts tests
```

## Citation

```bibtex
@software{phykn2026anchorconditioneddiffusion,
  author = {phykn},
  title = {Anchor-Conditioned Diffusion for Scalable 3D Microstructure Synthesis},
  year = {2026},
  url = {https://github.com/phykn/diffusion-gan3d}
}
```
