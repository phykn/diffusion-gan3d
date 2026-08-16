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
pixel loss lets exact boundaries adapt. Both losses and the connectivity loss
are disabled for samples whose anchor was removed by CFG
dropout. Coarse pooling cells are weighted by their observed coverage, so a
one-pixel partial-anchor edge cannot outweigh a fully observed cell. With
`model.anchor_multiscale: true`, the original observed phases and mask are
pooled independently at every encoder scale. Mask normalization preserves a
thin or partial plane, while zero-initialized projections preserve the original
generator path at initialization. Setting it to false selects the original
input-only path.

Before anchor training begins, a fixed EMA snapshot generates complete
anchor/VF-free 3D volumes for a frozen prior bank. During anchor training, the
connectivity critic compares at most seven three-slice windows: one containing
the anchor and up to two general windows from each of the three axes. Anchor and
general groups receive equal loss weight, so broader coverage does not dilute the
anchor boundary. Reference windows are sampled from the frozen volumes. A domain
uses its own prior for axes
it owns and falls back to provider-domain volumes only for a missing axis. After
the initial bank is complete, each domain periodically adds one volume from the
current EMA and evicts its oldest volume. This rolling update changes the prior
gradually without a second full-size bank or an abrupt global replacement.

The critic does not receive raw logits or raw slices. EMA references and student
outputs are converted to hard categorical phase images, and each three-slice
window becomes three bounded images: the first phase change, the second phase
change, and the change in those two changes. The student uses a straight-through
categorical conversion, so the critic sees the same representation on both sides
while gradients still reach the generator. This learns image-level continuation
without copying a teacher volume voxel by voxel. The existing normal-transition
loss remains as a small aggregate guardrail.

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
After each posterior update, the anchor trajectory is kept intact near the anchor
and tapered back to the baseline over a wide cosine window. This fixed coupling
preserves the same-RNG baseline exactly in the far field through the final step.
Context guidance is stronger early and reduces during cleanup, while plane
guidance rises toward the final step. The defaults (`anchor_strength=1`,
`anchor_sigma=2`) favor similar conditional structure over exact plane
reconstruction.
Script `03` reports
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
python scripts/03_check_anchor.py --weight run/<run-id>/generator.pt --domain 0 --gt scripts/gt_128.tiff --count 3 --anchor-strength 1 --anchor-sigma 2
python scripts/04_check_scale_up.py --weight run/<run-id>/generator.pt --domain 0
```

Generator scripts default to `--domain 0` and the `guidance` value in `config/gen.yaml`. Paper scripts require explicit weights and record input provenance.

`generator.pt` is the latest weight; `checkpoint_every_steps` preserves numbered
sets under `checkpoints/step_XXXXXXXX/`. Every `.pt` file is an independent model
`state_dict` containing tensors only. Optimizer state, training step metadata, and
trainer-side frozen prior volumes are never embedded in these weight files.

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
