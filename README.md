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
  domain_prob: 0.8
~~~

Folders listed under one axis are pooled within that domain. Training uniformly
samples one target domain per step. An axis present in the target domain uses
that domain's data. A missing axis uniformly borrows from domains that provide
that axis and removes the domain condition for that critic. `domain_prob` is the
probability of retaining the target-domain condition for a complete training
step; the remaining probability teaches the shared path without adding a common
domain ID.
Volume-fraction targets always use only the target domain. Anchors normally use
an owned axis; `anchor.cross_domain_prob` can instead present a
borrowed missing-axis section as an external/shared anchor. Incompatible
borrowed sections fall back to an owned anchor rather than borrowing another
domain's volume fraction.
Critic weight files follow the union of configured axes: if another domain
provides all three axes, all three critics are trained and saved; if the entire
configuration contains only axis 0, only `critic_0.pt` is created.

Anchor training separates appearance from continuation. A mask-normalized
coarse loss preserves every supplied section's phase layout, while a low-weight
pixel loss is applied only to the original measured section so its exact
boundaries can still adapt. Measured and EMA-derived coarse groups are averaged
separately, so adding more generated planes cannot drown out the measured root.
Both losses and the connectivity loss
are disabled for samples whose anchor was removed by CFG
dropout. Coarse pooling cells are weighted by their observed coverage, so a
one-pixel partial-anchor edge cannot outweigh a fully observed cell. Its pooling
scale is the power-of-two encoder scale nearest the geometric midpoint between
input and bottleneck resolutions; it therefore follows the model depth without
another setting. With
`anchor.multiscale_input: true`, the original observed phases and mask are
pooled independently at every encoder scale. Mask normalization preserves a
thin or partial plane, while zero-initialized projections preserve the original
generator path at initialization. Setting it to false selects the original
input-only path.

The real single-anchor path is learned first. Once its ramp is complete, one
fixed EMA snapshot generates a small bank of conditional 3D completions. Every
completion is rooted in a visible measured section, and that original section is
stored separately; VF is omitted so the bank captures continuation rather than a
particular composition target. The raw EMA volume remains the relation reference.
Only when constructing a multi-plane condition is the measured root overlaid on a
temporary copy, which keeps cross-axis intersections coherent without teaching an
artificial pasted seam as a real relation. Training then alternates real single anchors with
conditions assembled from one coherent bank volume. The latter always retain the
measured root and add a scale-balanced number of planes, possibly from different
axes. The maximum plane count and sampling stride follow the generator's encoder
downsampling factor, so this adds no tuning option.

With the usual one-root volume batch, the connectivity critic compares at most
seven three-slice windows. It reserves one available general window per axis and
always retains every measured-root window before sampling additional
generated-plane windows. Each generated plane keeps an exact three-slice source
relation from its conditional EMA completion. Its slice gap is sampled from the
EMA volume's chance-corrected correlation spectrum, so persistent structures are
also checked over wider ranges without a configured distance list. Fresh measured
roots and general windows, which have no aligned source, fall back to phase-fraction-matched prior
windows. Anchor and general groups receive equal loss weight, so using more
anchors does not increase the objective under the
usual one-root batch. A domain uses its own prior for axes it owns and falls back
to provider-domain volumes only for a missing axis. After the initial bank is
complete, each domain periodically adds one conditional completion from the
current EMA and evicts its oldest volume. This rolling update changes the prior
gradually without a second full-size bank or an abrupt global replacement.
Because section folders identify an axis but not a signed normal direction, the
connectivity critic always averages forward and reversed triplet scores.

The critic does not receive raw logits or raw slices. EMA references and student
outputs are converted to hard categorical phase images, and each three-slice
relation becomes three bounded images: the first phase change, the second phase
change, and the change in those two changes. The student uses a straight-through
categorical conversion, so the critic sees the same representation on both sides
while gradients still reach the generator. This learns image-level continuation
without copying a teacher volume voxel by voxel. Real-data 2D critics separately
guard marginal appearance, so an EMA completion contributes relations rather than
becoming the visual target. The existing normal-transition loss remains as a
small aggregate guardrail.

`condition_dropout.joint_each_prob` assigns the same probability to anchor-null,
VF-null, and joint-null states when both conditions exist. When VF is the only
condition, its null probability is derived as twice that value, preserving the
same marginal visibility without a second dropout setting.

Set `data.crop_partial: true` to train from a section whose height or width is
smaller than `data.crop_size`. Each available dimension is cropped to at most
`data.crop_size` and resized with the common
`data.input_size / data.crop_size` scale, so
the aspect ratio and physical voxel scale are preserved. The critic compares a
generated section window with the same rectangular shape. Each batch uses one
axis folder selected uniformly, regardless of its image count, so folders under
the same domain and axis may use different shapes. Omitting the option, or setting
it to `false`, keeps the strict error for
undersized images.
Physical resolution must remain consistent across the dataset.

To create a synthetic example dataset from [`config/simul.yaml`](config/simul.yaml):

```bash
python gen_data.py
```

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

Single-domain weights default to domain 0; multi-domain weights should specify a domain. Direct generation produces `data.input_size³`. Scale-up length per axis is `data.input_size + (blocks - 1) × (data.input_size - 2 × overlap)`, so 128-sized blocks with three blocks and overlap eight produce `352³`. Direct, anchor, and scale-up generation add one model downsampling cell beyond every outer face and return only the requested center, reducing zero-padding artifacts at the volume boundary. Low-level APIs and Script `04` retain an explicit `margin` override for controlled comparisons. `overlap` applies only to scale-up.

Shared generation defaults live in [`config/gen.yaml`](config/gen.yaml); CLI values override them:

```yaml
guidance: 1.0
overlap: 8
```

`anchor_strength=0` disables anchor conditioning. With anchors enabled, generation
keeps separate baseline and anchor trajectories from the same initial noise. Both
trajectories share every step's latent and posterior noise. All anchor planes are
passed in one joint conditional prediction, and its logit residual is blended
through a Gaussian spatial window before decoding. The Gaussian width is the
3D diagonal of one model downsampling cell, `sqrt(3) * downsample_factor`; the
current four-level model therefore uses approximately `13.86` voxels.
After each posterior update, the same normalized Gaussian also couples the anchor
trajectory back to the baseline. Coupling is one on the supplied plane and
decays with the same model-derived Gaussian width, so distant regions remain
baseline-dominant without a second pair of spatial-radius settings.
Context guidance follows the diffusion schedule: it starts at the natural
two-prediction residual scale $\sqrt{2}$ and tapers to the noise remaining at the
final reverse transition. Plane guidance rises toward the final step. The default
`anchor_strength=0.90` and model-derived spatial width spread the adaptation across
multiple slices instead of fitting the anchor through a narrow transition.
Script `03` reports slice-change rates for distances zero through 24 from the
nearest anchor. It also reports the first-difference change curve's second
difference, including its p95 and largest bend, to expose both distributed
roughness and isolated transition jumps. Script `03` always generates a same-RNG
unconditioned baseline and reports distance-wise voxel divergence from it. The
parameter-free effect summary reports divergence at the anchor, its mean over the
complete axis, and the farthest observed distance. Its
three passes are an unconditional reference, a separate conditioned trajectory,
and that trajectory's same-RNG baseline. `--seed` makes the full comparison
reproducible. `guidance=1` is the standard conditional path.
Validate non-default guidance with weights trained using condition dropout. See
[`PAPER.md`](PAPER.md) for details.


Run the diagnostic scripts with an explicit generator weight. Script `03`
first generates an unconditional reference volume, takes anchors from it, and
uses a separate random trajectory for the conditioned sample. Script `04`
requires a GT volume when `--count` is positive and `--anchor-strength` is
nonzero. Supplied GT TIFFs must be `uint8`, contain valid phase labels, and
exactly match the model's cubic input size; diagnostics never resize them:

```bash
python scripts/01_check_dataset.py
python scripts/02_check_generated.py --weight run/<run-id>/generator.pt --domain 0
python scripts/03_check_anchor.py --weight run/<run-id>/generator.pt --domain 0 --count 3
python scripts/04_check_scale_up.py --weight run/<run-id>/generator.pt --domain 0
python scripts/05_check_continuation.py --weight run/<run-id>/generator.pt --no-view
```

Script `05` first generates an unconditioned reference, takes its boundary
section, and uses that one section to generate the remaining 3D volume jointly.
Passing `--anchor` instead uses the center crop of a real 2D label image. The
script reports boundary agreement, first-plane continuation, slice flicker, and
same-RNG baseline drift. Use `--out` for the generated TIFF and `--figure` for a
non-interactive distance-wise slice strip. Use `--napari` to inspect the full
generated volume in 3D; the condition input remains separately labeled in the
Matplotlib slice strip.

Rebuild the paper's tracked data and figures from one explicit checkpoint in
this order:

```bash
python scripts/paper/evaluate_structure.py --weight run/<run-id>/generator.pt
python scripts/paper/make_assets.py --reference temp/paper_structure/direct_seed_0.tiff
python scripts/paper/make_anchor_asset.py --weight run/<run-id>/generator.pt
python scripts/paper/evaluate_continuation.py --weight run/<run-id>/generator.pt
python scripts/paper/make_scale_up_asset.py --weight run/<run-id>/generator.pt
```

The two evaluation scripts record the complete raw rows, summaries, weight hash,
training-config hash, and generation settings in tracked JSON sidecars. Paper
figures no longer depend on a separately maintained `gt.tiff`.

Generator scripts default to `--domain 0`. Conditional Scripts `03` and `04` use the `guidance` value in `config/gen.yaml`; Script `02` has no anchor or phase-fraction condition, so it exposes no guidance option. Paper scripts require explicit weights and record input provenance.

`generator.pt` is the latest weight; `train.archive_every` preserves numbered
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
