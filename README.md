# Anchor-Conditioned Diffusion for Scalable 3D Microstructure Synthesis

Three-dimensional imaging is expensive, so material structures are often known only through a few measured 2D sections. Existing generators can produce plausible 3D volumes, but they do not preserve a section at its measured location. Pasting it into the result afterward can also create an artificial discontinuity.

This project uses measured sections as anchors throughout denoising, preserving their phases while generating a connected 3D structure around them. The anchored volume can then guide jointly generated overlapping blocks. During scale-up, the base is softly conditioned at every reverse step and remains free to adapt to its new surroundings; it is not pasted back into the final volume.

[`PAPER.md`](PAPER.md) records the evaluated method and its limitations. Its
status note lists the material differences from the current training and
scale-up implementation.

## Installation

```bash
git clone https://github.com/phykn/diffusion-gan3d.git
cd diffusion-gan3d
pip install -r requirements.txt
```

## Usage

Set the 2D section folders and training options in [`config/train.yaml`](config/train.yaml), then train the model:

```bash
python run_train.py --device cuda
# Or choose the exact output directory (it must not already exist):
python run_train.py --device cuda --run-dir run/my-experiment
```

The `data.augment` preset controls differentiable critic augmentation for both
real and generated slices. Use `isotropic` when all three axes are equivalent,
`transverse_0`, `transverse_1`, or `transverse_2` when one axis is distinct,
`directional` when axis identities must be preserved without 90-degree swaps,
or `false` to disable it. The configuration file keeps the same choices in a
comment above the data section.

Training configuration schema `2` intentionally starts a new run rather than
loading pre-schema checkpoints. Anchor-task sampling is separate from
classifier-free condition dropout: with both anchor and volume-fraction (VF)
conditions available, the anchor-null, VF-null, and joint-null states each use
5% of samples; a lone VF condition is dropped on 10%. VF targets are sampled
from individual real crops instead of averaging the full axis batch, and the
loss measures total-variation distance after exact anchor projection.

Two auxiliary terms target the failure modes exposed by scale-up. The
anchor-normal term matches center-to-neighbor phase transitions to matched
unconditional replay triplets. Scale consistency uses the EMA model as a stop-gradient teacher
and matches the predicted clean probabilities in the 16-voxel region shared by
adjacent 128-core, 8-halo views. Its 25% sampling probability starts after 1,000
steps and ramps for 4,000 steps. This probability is conditional on a sampled
training volume being large enough for the 144-voxel consistency view; with the
default five volume sizes, three are eligible, so the steady-state rate is 15%
of all steps. The configured `0.10` normal-transition weight
and `1.0` scale-consistency weight are conservative starting values, not
universal constants; compare seeded ablations before choosing final weights.

Generate a 3D volume around a known section and extend it to `3 × 3 × 3` blocks:

```python
from src.anchor import PlaneAnchor
from src.build import load_generator
from src.scale import ScaledGenerator

generator = load_generator(generator_path, device)

anchor = PlaneAnchor(image, axis=0, index=32)
base = generator.generate(anchors=(anchor,))

volume = ScaledGenerator(generator).generate(
    blocks=(3, 3, 3),
    overlap=8,
    base=base,
)
```

At the default `anchor_strength=1`, known voxels follow the matching DDPM
forward/reverse bridge at every transition and are exact in the final base.
Values between zero and one relax that projection by blending it with the model
prediction; zero leaves the unconditioned sampling path unchanged. Exact anchor
values do not by themselves guarantee a smooth normal-direction continuation,
so the anchor diagnostic reports boundary change, transition, and continuation
metrics separately from anchor accuracy.

The samplers also accept `guidance_scale` for classifier-free guidance in
logit space. The default `1.0` follows the single-pass conditional path exactly.
Values above one extrapolate along the learned conditional-logit direction and
may increase condition influence; zero uses one unconditional evaluation.
`guidance_scale=0` disables only the learned condition inputs, not anchor
projection; use `anchor_strength=0` to remove the anchor. Non-default guidance
requires weights trained with condition dropout. Scales other than zero or one
use two denoiser evaluations whenever a learned condition is active, so select
the value with a seeded quality sweep rather than assuming that larger is
better. Hard anchors remain exact because projection is applied after the
guided prediction. For `ScaledGenerator`, guidance affects the learned `vf`
condition; the external `base` uses a separate spatial blend and is not amplified
by this option.

Run the diagnostic scripts with an explicit generator weight. Script `03`
always requires a GT volume; script `04` requires one when `--count` is positive
and `--anchor-strength` is nonzero:

```bash
python scripts/01_check_dataset.py
python scripts/02_check_generated.py --weight run/<run-id>/generator.pt
python scripts/03_check_anchor.py --weight run/<run-id>/generator.pt --gt scripts/gt_128.tiff --count 3
python scripts/03_check_anchor.py --weight run/<run-id>/generator.pt --gt scripts/gt_128.tiff --count 3 --anchor-strength 0.65
python scripts/04_check_scale_up.py --weight run/<run-id>/generator.pt
python scripts/04_check_scale_up.py --weight run/<run-id>/generator.pt --gt scripts/gt_128.tiff --count 3
```

These generation and evaluation CLIs accept `--guidance-scale` with a default
of `1.0`. Generator-invoking paper CLIs also require an explicit `--weight`;
they never select the newest run implicitly, so cached metrics and assets
remain tied to a known checkpoint.

Paper manifests record resolved paths and SHA-256 hashes for the checkpoint,
training configuration, reference, and any additional explicit inputs, along
with guidance and generation arguments/signature. Cached volumes and derived
outputs are recorded as resolved paths; reuse requires the exact path set to
exist. Output paths are rejected when they alias an explicit input. Prefer an
immutable numbered checkpoint for long evaluations.

The `gt` argument supplied to script `03`, or to script `04` with active
anchors, is the reference volume from which anchor planes are selected.

`save_every_steps` updates the latest `generator.pt` and critic files. The
independent `checkpoint_every_steps` interval preserves complete numbered sets
under `checkpoints/step_XXXXXXXX/`, so intermediate weights are not overwritten.

## Code layout

- `src/generate.py` owns direct patch-size sampling; `src/scale.py` owns tiled
  planning, storage, overlap fusion, and large-volume sampling.
- `src/config.py` owns the schema and run-config discovery contract;
  `src/build.py` is the composition root that creates training and inference
  objects.
- `src/train/engine.py` owns one optimization step and its losses;
  `src/train/runner.py` owns progress reporting, TensorBoard output, and
  checkpoint cadence.
- `scripts/` contains user diagnostics, while `scripts/paper/` contains
  provenance-checked evaluation and figure generation.
- `tests/` contains the tracked regression suite.

For development, install the runtime and test tools together and run:

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
