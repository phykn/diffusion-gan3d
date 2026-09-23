# Diffusion-GAN 3D

Generate 3D microstructures from 2D label images, with optional anchor planes,
overlapping-tile generation, and a separate super-resolution stage.
Training uses real 2D sections without measured 3D targets.

LR volume-fraction conditions are computed per generated sample from its measured
anchor, including the measured root of an anchor replay, or from one real crop
when no anchor is selected. Height-conditioned crops share their VF and height
coordinates; an enabled spatial profile supplies the VF through its mean.
These 2D fractions are conditioning targets, not measured 3D volume fractions.

The default LR preset matches phase-pair statistics measured in real 2D sections
at the final reverse step. For each observed physical direction and separation
of 1 through `loss.connectivity.max_slice_gap` LR cells, it compares the joint
phase probabilities; their diagonal measures same-phase persistence. Phase
fractions remain continuous and gradients flow through generated samples.
One LR cell corresponds to `crop_size / lo_res_size` source pixels. Only the
target domain's own planes supply these statistics, before critic augmentation.
With height conditioning, comparisons use overlapping normalized height intervals;
xy observations without known height are excluded. Missing directions or height
overlap provide no target. Logs under `real_transition/` report errors and matched
observations; `loss/real_transition` reports their averaged loss.

Anchor replay still supplies multi-anchor conditions. Its outputs are not targets
for the default continuity loss, and its connectivity critic is inactive. The
legacy replay losses remain opt-in for controlled comparisons:

| Mode | `adversarial_weight` | `normal_transition_weight` | `real_transition_weight` |
| --- | ---: | ---: | ---: |
| Legacy replay | 0.25 | 0.1 | 0 |
| No continuity loss | 0 | 0 | 0 |
| Measured transitions (default) | 0 | 0 | 0.1 |

These keys are under `loss.connectivity`. Compare separate runs with the same
data, initialization, anchor/VF settings, and schedule. Check anchor-boundary
jumps, held-out 2D transition statistics, and sample diversity. Matching these
statistics does not establish 3D connectivity. Saved configurations retain their
loss settings on resume; changing the objective requires a separate training run.

SR samples a spatial rotation/reflection independently for each LR bank volume
before coarse-input corruption and resizing. Cubic volumes use all 48 cube
symmetries, including identity; height-conditioned volumes use only the eight
xy symmetries that preserve z coordinates. The transformed clean volume supplies
the consistency target, while its corrupted version supplies the model condition.
Phase fractions and bank snapshots are preserved. This volume augmentation is
separate from the configured 2D critic augmentation.

## Setup

Use Python 3.11+ and a PyTorch build suitable for your CUDA environment.
Run commands from the project root in the activated project `.venv`.

```bash
git clone https://github.com/phykn/diffusion-gan3d.git
cd diffusion-gan3d
python -m venv .venv
# Activate .venv, then install dependencies:
python -m pip install -r requirements.txt
```

In PowerShell, use the interpreter directly without activation:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For development, install the test tools in the same environment. `httpx` is
required by the backend tests:

```powershell
& .\.venv\Scripts\python.exe -m pip install pytest ruff httpx
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m ruff check .
```

Frontend checks run from `frontend/`: `npm ci`, `npm test`, and `npm run build`.
Tests use a fixed available-RAM probe; dedicated memory tests supply their own
limits. CUDA tests are skipped when CUDA is unavailable.

## Train

Set image folders, phase counts, and resolutions in
[`config/data/default.yaml`](config/data/default.yaml). Inputs are 2D images
containing integer phase IDs (0–255, with `num_phases` between 1 and 256);
plane names are `xy`, `xz`, and `yz`.
Each training image must contain exactly one 2D frame; split multi-page TIFF
stacks into individual sections first. Generator channel widths must be at least
2 because single-channel normalization discards the input signal.
Edit the [LR](config/train/low_res.yaml) and [SR](config/train/sr.yaml) training
presets as needed.

Datasets always return a dict containing `image` and source/crop metadata.
With `conditioning.height_enabled: true`, height is fixed to z: the vertical
direction of `xz`/`yz` section images in source pixels. Augmentation and the
connectivity critic automatically preserve this height direction when height
conditioning is enabled, so there is no separate axis setting. Source-image
heights are saved for inference. `xy` sections have no measured z coordinate
(`height_origin` and `height_extent` are -1).

```bash
python run_train_1st.py --device cuda
python run_train_2nd.py --base-weights "run/my-lr-run" --device cuda
```

SR trains from a frozen LR model and real HR sections. Weights, resolved settings,
and metrics are saved under `run/`.

Resume from `checkpoints/last.pt` with `--resume`; `--steps` is the total target,
including completed steps. Each resume writes to a new run directory.
If files moved to another computer, copy the original images and run artifacts,
then map their old path prefixes to their new locations:

```powershell
& .\.venv\Scripts\python.exe run_train_1st.py --resume "D:/project/run/my-lr-run/checkpoints/last.pt" --path-map "C:/project" "D:/project" --steps 20000 --device cuda
```

The same `--path-map OLD NEW` option works for `run_train_2nd.py`; repeat it for
separate image and run roots. SR also needs its saved LR bank and frozen LR
`generator.pt`/`train.yaml`. Keep those files unchanged: path mapping preserves
image, bank, and source hashes and all training settings. Mappings persist in
new checkpoints, including for height-conditioned bank refreshes; provide new
mappings only when paths move again.

SR bank snapshots publish atomically and never replace an existing step. Store
runs on a filesystem with hard-link support (such as NTFS or ext4).

### Critic input comparison

New LR and SR training presets use `model.critic.input_mode: single`: only x_t
enters the plane critic, retaining time/domain/height/profile conditions. Its
input has num_phases channels and its forward method has no x_{t+1} argument.
`pair` remains available for legacy checkpoints and controlled comparisons.
Historical configurations without input_mode still resolve to pair for resume
compatibility; use the updated presets or explicitly set single for new training.
The new preset removes a discrimination path through a detached state whose
real/fake distributions can differ. This is a design choice, not a demonstrated
quality improvement. Posterior sampling links adjacent states algebraically;
it does not by itself guarantee the learned reverse conditional is correct.
The generator still receives x_{t+1} in both modes. Single-input critics remove
the detached-current-state discrimination path but no longer score the joint
diffusion transition. This is a marginal critic ablation, not a reproduction of
the Diffusion-GAN training algorithm. LR and SR both support the setting;
switching modes requires a fresh critic and optimizer, not checkpoint resume.

For paired LR experiments, use a resolved training configuration with
`train.num_workers: 0` and no `train.initial_weights`:

```powershell
& .\.venv\Scripts\python.exe scripts/experiments/critic_inputs.py --config config/train/low_res.yaml --run-dir run/critic-input-comparison --seeds 11 22 33 --steps 10000 --device cuda
```

The runner refuses an existing output directory, checks identical generator
initialization hashes, seeds sampling per step, and trains fresh critics for
both modes. It retains all other settings, saves each run's configuration,
checkpoints and metrics, and summarizes time, CUDA peak allocation and per-time
input-gradient diagnostics in `report.json`. Timings include checkpoint writes.
Large current-input sensitivity alone does not prove a shortcut. On held-out
data, a separately trained current-only probe and within-condition pair
shuffling can provide additional evidence; this runner does not perform those
probe experiments or claim a quality ranking. Compare held-out directional
statistics, diversity, VF/height/anchor compliance and boundary discontinuities
before selecting a default, then validate SR consistency separately. Split by
source image or volume rather than neighboring patches to avoid leakage.

## Generate and inspect

Pass a run directory or a `generator.pt` file:

```bash
python scripts/02_check_generated.py --weight "run/my-lr-run"
python scripts/03_check_anchor.py --weight "run/my-lr-run"
python scripts/04_check_scale_up.py --weight "run/my-lr-run"
python scripts/06_check_hr.py --weight "run/my-sr-run"
```

Results are saved under `run/checks/`. Add `--no-view` to save without opening a
viewer. See the [manual-check guide](docs/checks.md) for all six scripts.

`anchor_strength` ranges from 0 (no anchor) to 1 (full anchor conditioning).
Intermediate values interpolate the logits of the anchor-present and
anchor-absent paths at the same diffusion state and latent, retaining VF, profile,
domain, and height conditions. The model always receives a binary anchor mask.
CFG is applied to the interpolated logits. Intermediate strength requires two
model evaluations per transition, or three when non-unit CFG also conditions on
VF or a profile; zero CFG needs only its unconditional path. Existing weights
remain loadable, but the updated per-sample VF training requires further training
to affect existing models. Strength does not specify an exact pixel-match rate.

Height-conditioned inference uses the saved side-image height and starts at
`height_origin=0` by default. If source heights differ within a domain, pass the
intended source's `height_extent`; use `height_origin` for an offset crop.
Both values use original image pixels, and the requested volume must fit inside
that height.

## Web interface

Start the backend, then run the frontend in another terminal (Node.js required):

```bash
python backend/run.py --weight "run/my-lr-run"
```

```bash
cd frontend
npm ci
npm run dev
```

Server limits are configured in [`backend/config.yaml`](backend/config.yaml).
Select the section plane and, for multi-domain models, the domain in the sidebar.
Height-conditioned xz/yz inputs use the crop row as their Z origin and the full
uploaded image height as their extent. XY inputs require separate Z coordinates.
Anchors start at the output origin, including when generating multiple blocks.

## Project layout

| Path | Purpose |
| --- | --- |
| `src/api.py` | Public inference APIs, plane anchors, HR extension, and app factory |
| `src/config/`, `config/` | Configuration loading, validation, defaults, and presets |
| `src/data/` | Image sources, datasets, batch streams, augmentation, and slice sampling |
| `src/prepare/` | Phase-fraction resizing and physical height/profile coordinates |
| `src/model/`, `src/build/` | Neural networks and diffusion; model/data/trainer assembly |
| `src/train/trainer.py`, `src/train/loss/` | Training steps, optimizer updates, and losses |
| `src/train/batch.py` | Explicit step inputs, sampled pairs, and their coordinates/conditions |
| `src/train/state.py`, `src/train/coarse.py` | LR/SR checkpoint state; coarse-input corruption |
| `src/train/run/` | Run setup, LR bank generation, logging, and checkpoint scheduling |
| `src/predict/` | Inference, volume conversion, and memory estimates |
| `src/predict/tiling/`, `src/predict/sr/` | Overlapping-tile sampling and super-resolution |
| `src/evaluate/` | Label, slice, volume, connectivity, and structure measurements |
| `src/anchor.py`, `src/plane.py`, `src/storage.py` | Anchor encoding, plane conventions, and artifact I/O |
| `scripts/` | Numbered inspections, shared CLI/display helpers, experiments, and paper tools |
| `backend/`, `frontend/` | HTTP service and web UI |
| `simul/` | Synthetic data generation; run `python simul/run.py` |
| `tests/`, `frontend/test/` | Python and frontend regression tests |
| `run/` | Local weights and generated outputs |

Training commands enter `src/train/run/`, which uses `src/build/` to assemble the
trainer with its settings and data metadata. Each real batch carries its images,
domains, height coordinates, profiles, and source geometry through the training
step. `Trainer` separates LR/SR batch preparation and owns optimizer updates;
`src/train/loss/denoiser.py` computes the differentiable generator objective.
SR bank creation and refresh live in `src/train/run/bank.py`; shared frozen-source
validation and configuration loading live in `src/train/run/source.py`. The
runner publishes a snapshot after preparation succeeds.

The web UI calls the backend, which delegates generation to the inference APIs.
Model weights and training checkpoints have separate formats; keep
`generator.pt` for inference and `checkpoints/last.pt` for resume.

[Refactoring decisions and verification](docs/refactoring.md) records the module
boundaries, corrected defects and remaining validation limits.

[Method and recorded experiments](PAPER.md) · [MIT License](LICENSE)
