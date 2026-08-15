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

Each numeric domain contains folders for one or more axes. Training creates and
updates critics only for axes that appear anywhere in the configuration, so a
single axis is sufficient:

~~~yaml
data:
  domains:
    0:
      0: [data/domain_0/axis_0]
    1:
      0: [data/domain_1/axis_0]
      1: [data/domain_1/axis_1]
      2: [data/domain_1/axis_2]
  domain_dropout: 0.2
~~~

Folders listed under one axis are pooled within that domain. Training uniformly
samples one target domain per step. An axis present in the target domain uses
that domain's data. A missing axis uniformly borrows from domains that provide
that axis and removes the domain condition for that critic. `domain_dropout`
also removes the domain condition from complete training steps with the given
probability, teaching the shared network path without adding a common domain ID.
Volume-fraction targets always use only the target domain. Anchors normally use
an owned axis; a bounded `anchor.shared_axis_probability` can instead present a
borrowed missing-axis section as an external/shared anchor. Incompatible
borrowed sections fall back to an owned anchor rather than borrowing another
domain's volume fraction.
Critic weight files follow the union of configured axes: if another domain
provides all three axes, all three critics are trained and saved; if the entire
configuration contains only axis 0, only `critic_0.pt` is created.

Anchor training separates appearance from continuation. A mask-normalized
coarse loss preserves the supplied section's phase layout, while a low-weight
pixel loss lets exact boundaries adapt. Both losses, the connectivity loss, and
teacher promotion are disabled for samples whose anchor was removed by CFG
dropout. The optional relation loss stores only phase-relation curves from a
separate anchor-free, full EMA diffusion sample and morphology descriptors—
never generated images. It measures chance-corrected
center-to-slice phase relations at every available distance, learns the useful
distance range from the stored curves, and stops gradients through the anchor
plane. Domain curves are used only for axes actually observed in that domain;
otherwise training falls back to a balanced shared bank built from domains that
own the axis. An out-of-distribution descriptor gate disables unsuitable shared
matches instead of forcing an external anchor toward a generated style.

The repository defaults keep pseudo multi-anchor replay off while this relation
path is trained. Existing model checkpoints remain load-compatible because the
new banks are trainer-side statistics rather than model parameters, but weights
must be fine-tuned or retrained to learn the new objective.

Set `data.allow_partial_crop: true` to train from a section whose height or
width is smaller than `crop_size`. Each available dimension is cropped to at
most `crop_size` and resized with the common `input_size / crop_size` scale, so
the aspect ratio and physical voxel scale are preserved. The critic compares a
    generated section window with the same rectangular shape. Each batch uses one
    axis folder selected uniformly, regardless of its image count, so folders under
    the same domain and axis may use different shapes. Omitting the option, or setting
it to `false`, keeps the strict error for
undersized images.
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
    margin=8,
    base=base,
    domain=0,
)
```

Single-domain weights default to domain 0; multi-domain weights should specify a domain. Direct generation produces `input_size³`. Scale-up length per axis is `input_size + (blocks - 1) × (input_size - 2 × overlap)`, so 128-sized blocks with three blocks and overlap eight produce `352³`. Direct, anchor, and scale-up generation add the configured margin beyond every outer face and return only the requested center, reducing zero-padding artifacts at the volume boundary. `overlap` applies only to scale-up.

Shared generation defaults live in [`config/gen.yaml`](config/gen.yaml); CLI values override them:

```yaml
guidance: 1.0
overlap: 8
margin: 8
```

`anchor_strength=0` disables anchor conditioning. With anchors enabled, generation
keeps separate baseline and anchor trajectories from the same initial noise. Both
trajectories share every step's latent and posterior noise. All anchor planes are
passed in one joint conditional prediction, and its logit residual is blended
through a Gaussian spatial window controlled by `anchor_sigma` before decoding.
The residual is smoothed along each anchor's normal axis by the separate
`anchor_residual_blur` control. By default, this blur decreases from
`anchor_residual_blur_early=2` at the first noisy steps to
`anchor_residual_blur=1.5` at the final cleanup steps. After each posterior update,
the anchor trajectory is kept intact near the anchor and tapered back to the
baseline over a wide cosine window. This prevents distant structure from drifting;
the coupling is gradually released during final cleanup so the model can resolve
the transition instead of leaving a hard mixture. The defaults (`anchor_strength=1`,
`anchor_sigma=2`,
`anchor_residual_blur_early=2`, `anchor_residual_blur=1.5`) favor
similar conditional structure over exact plane reconstruction. Script `03` reports
slice-change rates for distances zero through 24 from the nearest anchor to make
displaced transition seams visible. `guidance=1` is the standard conditional path;
pass `--compare-unconditioned` to Script `03` to report distance-wise voxel
divergence from a same-RNG unconditioned generation. This adds one generation pass.
validate non-default guidance with weights trained using condition dropout. See
[`PAPER.md`](PAPER.md) for details.


Run the diagnostic scripts with an explicit generator weight. Script `03`
always requires a GT volume; script `04` requires one when `--count` is positive
and `--anchor-strength` is nonzero:

```bash
python scripts/01_check_dataset.py
python scripts/02_check_generated.py --weight run/<run-id>/generator.pt --domain 0
python scripts/03_check_anchor.py --weight run/<run-id>/generator.pt --domain 0 --gt scripts/gt_128.tiff --count 3 --anchor-strength 1 --anchor-sigma 2 --anchor-residual-blur-early 2 --anchor-residual-blur 1.5
python scripts/04_check_scale_up.py --weight run/<run-id>/generator.pt --domain 0
```

Generator scripts default to `--domain 0` and the `guidance` value in `config/gen.yaml`. Paper scripts require explicit weights and record input provenance.

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
