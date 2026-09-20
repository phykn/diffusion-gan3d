# Anchor-Conditioned Diffusion for Scalable 3D Microstructure Synthesis

수동 확인은 [웨이트만 지정하는 실행표](docs/checks.md)를 참고하세요. `scripts/01~06`이 기본 진입점이며 결과는 `run/checks/`에 자동 저장됩니다.

Generate natural 3D microstructures from 2D sections. The model supports:

- unconditional and domain-conditioned generation
- generation around a supplied 2D anchor
- large-volume generation with overlapping blocks
- a separate 3D super-resolution stage trained from generated LR volumes and real HR 2D sections

Unlike post-generation pasting, anchor conditioning is applied throughout
denoising so the surrounding 3D structure can adapt naturally. See
[`PAPER.md`](PAPER.md) for the method and evaluation.

## Install

Use Python 3.11 or newer and the project's own virtual environment. Install the
PyTorch build appropriate to the machine's CUDA environment.

```bash
git clone https://github.com/phykn/diffusion-gan3d.git
cd diffusion-gan3d
pip install -r requirements.txt
```

## Resolution and training

Training settings are organized by responsibility:

```text
config/
  data/default.yaml     # image folders, phase count and resolution
  train/low_res.yaml    # stage-1 model and training
  train/sr.yaml         # SR model and training
  gen.yaml              # stage-1 inference defaults
```

Both training presets select `data: config/data/default.yaml`. To use another
dataset, copy that data preset and select it with `--data config/data/battery.yaml`.
Data-file references and image paths in those presets resolve from the project
root, independently of the training YAML's location. Absolute paths are accepted.
The data YAML contains the data fields directly, without an outer `data:` key.

Training presets show the settings normally adjusted for a dataset or experiment.
Advanced options may be omitted: `TRAIN_DEFAULTS` and `STAGE_DEFAULTS` in
[`src/config/defaults.py`](src/config/defaults.py) define their configuration defaults in one place.
Loading rejects unknown and obsolete keys with their full configuration path,
then resolves defaults without overwriting explicit values. Each run saves the resolved settings, including omitted options;
reloading that snapshot retains its values even if the code defaults later change.

| Treatment | Settings |
|---|---|
| Calculated from other inputs | HR grid from LR size × SR scale; domain count from data; critic count from plane groups |
| Optional fallback | `optim.critic_lr` follows `optim.generator_lr` when omitted |
| Advanced defaults | Embedding/latent/noise channels, diffusion beta limits, anchor schedule and dropout, local critic weighting and R1 interval, SR downsampling tolerance, Adam betas, EMA, precision and loader options |
| Explicit experiment choices | Image/phase meanings, crop and LR sizes, SR scale, network widths, plane groups and augmentation, loss weights, batch/slice counts, training duration and saving intervals |

To override an advanced option, add its original key to the appropriate existing
section; there is no separate advanced YAML or inheritance chain. For example:

```yaml
optim:
  generator_lr: 0.0001
  critic_lr: 0.0002
  ema_decay: 0.995

train:
  total_steps: 20000
  mixed_precision: false
```

This is a partial override example, not a complete training preset. In a saved
snapshot both learning rates are explicit; changing only `generator_lr` there
keeps the saved critic rate. Delete `critic_lr` to select the fallback again.
The standard LR preset intentionally keeps different generator/critic rates;
the SR preset omits its duplicate critic rate. Batch and slice counts are
independent sampling controls, and loss weights are independent objectives;
neither is derived from matching current values. Augmentation is not inferred
from critic sharing or thickness direction.

| Training group | What to change here |
|---|---|
| `model` | Generator, critic and diffusion architecture |
| `augmentation` | Training transformations and their probability |
| `conditioning` | Domain, anchor and volume-fraction selection (LR); coarse corruption (SR) |
| `loss` | Training objectives and regularization weights |
| `optim` | Learning rates, Adam betas and EMA decay |
| `lr_bank` | LR sample count, generation guidance and refresh interval (SR) |
| `train` | Batch sizes, update counts, precision and saving intervals |

Lists use `[a, b]`; blank lines separate related groups. Probabilities are in
`[0, 1]`, and `*_every_steps` values count training steps. `real_batch_size` counts
2D images per plane; `volume_batch_size` counts 3D volumes. In both stages,
`slice_pairs_per_plane` counts pairs of adjacent diffusion times. Each active
critic group and the generator update once per step. `downsample_*` refers to reducing SR output to its LR input
grid; `lr` in `generator_lr`/`critic_lr` means learning rate.

Stage 1's `borrowed_plane_probability` selects an anchor from a plane absent in
the target domain and supplied by another domain. Only use it when that sharing
assumption is appropriate. `dropout_probability_per_case` assigns the same
probability to dropping both anchor and volume fraction, anchor alone, or volume
fraction alone; when no anchor exists the volume-fraction drop probability is
twice this value. These mechanisms are unchanged by the config reorganization.

The data preset defines the observed field and both model grids:

```yaml
crop_size: 128
lo_res_size: 64
hi_res_size: 128
```

`crop_size` is the square region read from the original image. `lo_res_size` is
the stage-1 grid and `hi_res_size` is the stage-2 grid over that same field of view.
The SR scale is their ratio; there is no separately declared `scale_factor` key.
The HR grid need not equal the original crop size. Stage 1 ignores the HR grid.

| Original crop | LR size | Scale | HR size | Generation |
|---|---|---|---|---|
| 128 | 64 | 2 | 128 | 64³ → 128³ |
| 256 | 64 | 1.5 | 96 | 64³ → 96³ |
| 256 | 64 | 4 | 256 | 64³ → 256³ |

Both stages describe the same original field of view. New stage-1 training and
anchor preparation use **original crop → LR phase map** directly. SR uses
**original crop → HR phase map** for its real 2D sections. Changing the SR scale
does not change stage-1 preprocessing or require retraining the LR model.
Phase-channel area averaging is used for reduction, retaining fractional occupancy
in real sections, anchor conditions and volume-fraction targets. Label IDs are never
numerically averaged. Final exported volumes use integer phase IDs. Retaining
fractions does not guarantee that a subvoxel connection survives final discretization. Larger target grids use interpolated phase
channels; enlarging the original data does not supply additional measured detail.
Fractional scale factors are supported when the resulting grid size is integral.
Partial crops are rejected to keep the physical field of view consistent.
Only the current configuration schema is accepted. Old `input_size`,
`data.scale_factor`, `model.generator.scale_factor`, numeric configuration planes, global `augmentation.mode`,
`train.seed` and `train.stability_version` are rejected; no migration runs.
Existing checkpoints and old configuration files are unsupported. Retrain using
the current presets.

### Stage 1: low-resolution 3D

Set the 2D label-image folders in
[`config/data/default.yaml`](config/data/default.yaml) and training options in
[`config/train/low_res.yaml`](config/train/low_res.yaml), then run:

```bash
python run_train_1st.py --device cuda
# Another dataset, with the same training recipe:
python run_train_1st.py --data config/data/battery.yaml --device cuda
```

The default presets use `data/sample.png` as an isotropic two-phase sample for
all three planes, with one shared slice critic in each stage. They use 128-pixel
crops, a 64³ first-stage grid and 128³ second-stage diffusion refinement.
Height conditioning is disabled; flips and quarter-turn rotations are enabled
in all planes. Data paths select folders nonrecursively, so additional images
placed directly in `data/` will also enter training.

On Windows, launch `run_train_1st.bat` using the project `.venv`. After stage 1
has saved weights, set `source.weights` in `config/train/sr.yaml` to that run
folder or its `generator.pt` path, then launch `run_train_2nd.bat`. Both batch
files also forward CLI arguments, including `--resume` and `--base-weights`.

Name the training planes `xy`, `xz`, and `yz`:

```yaml
domains:
  0:
    xy: [data]
    xz: [data]
    yz: [data]
```

Volumes retain their D,H,W array order, interpreted as z,y,x. `xy` is normal
to axis 0 (z), `xz` to axis 1 (y), and `yz` to axis 2 (x). Raw slice row/column
order is y/x, z/x, and z/y respectively. This naming does not rotate or flip
existing images; acquisition orientation must match the corresponding plane.
Configuration planes must use these names. The numeric `PlaneAnchor` axis
interface remains available.

Set `model.critic.plane_groups` independently in each training preset. Each inner
list creates one critic and one optimizer; its planes share the model weights.

| Slice critics | Configuration | Assumption |
|---|---|---|
| 1 (default) | `plane_groups: [[xy, xz, yz]]` | All three plane distributions can share a critic |
| 2 | `plane_groups: [[xy], [xz, yz]]` | The two thickness sections can share a critic |
| 3 | `plane_groups: [[xy], [xz], [yz]]` | Each plane is modeled separately |

These examples assume all three planes are present in `data.domains`. List every
observed plane exactly once across groups; omit planes with no observations.
Sharing is an explicit modeling assumption, never inferred from matching image
paths. A shared group does not authorize rotations or supply missing plane data.
The separate connectivity critic is not included in these counts.

Losses are averaged over member planes and then critic groups in both stages.
Stage-1 critics remain domain conditioned; SR keeps groups separate per domain.
Group names use canonical plane order: `critic_xy.pt`, `critic_xz_yz.pt` and the
separate `critic_c.pt`. SR names include the domain, such as `0_xz_yz`.
`plane_groups` must be explicit and cannot change on resume.

Training checkpoints keep one optimizer per critic and no RNG state. Artifact
tags identify LR training, SR training, SR inference weights and fractional LR
banks without version numbers. Only the current schemas are supported; there
are no legacy compatibility paths.

### Plane orientation and augmentation

For a substrate-to-surface direction along z, set `thickness_axis: z` in the data
preset and orient the input images so increasing z means substrate to surface.
The raw row/column convention is xy: y/x, xz: z/x, yz: z/y. Neither the data
loader nor the critic groups infer or correct image orientation.

Both training presets use the following explicit plane policies:

```yaml
augmentation:
  probability: 0.5

  planes:
    xy:
      flip_axes: [x, y]
      rotate_90: true

    xz:
      flip_axes: [x]
      rotate_90: false

    yz:
      flip_axes: [y]
      rotate_90: false
```

`flip_axes` names physical directions, not array dimensions. Here thickness
sections can flip horizontally but keep substrate and surface in place. The xy
policy additionally assumes in-plane reflection and quarter-turn symmetry;
disable transformations that do not match the material. Use `flip_axes: []` and
`rotate_90: false` to leave a plane unchanged. Policies stay independent even
when those planes share a critic.

`data.thickness_axis` rejects flips of that axis and quarter-turn rotations in
planes containing it, and disables connectivity reversal along the thickness normal.

Set `conditioning.height_enabled: true` in both stages for nonuniform thickness.
The loader retains crop origins and derives `data.height_extents` per domain from
full-thickness side images (the row or column containing the thickness direction).
Each side image supplies its own full height and crop origin; all images must share
the same top-to-bottom interpretation. Heights may differ. The saved domain default
is the unique source height, or null when it is ambiguous. In that case inference
requires height_extent (--height-extent in CLI), in original image pixels.
`height_origin` in Python/HTTP and `--height-origin` in the CLI specify the output
origin in source-image pixels. LR, SR and tiles use the same cell-center coordinates.
Side-plane critics receive aligned coordinate fields; sections normal to the
thickness axis have unknown absolute positions and remain marginal comparisons.
Keep the option false for homogeneous material or images with unknown cropping.

For explicit policies, `probability` is the probability of selecting an allowed
non-identity transform. Quarter turns include their compositions; rectangular
inputs exclude transforms that exchange height and width. Each diffusion-time
pair and all channels supplied together receive the same transform, including
connectivity triplets. SR also couples the transforms of its real/fake slice
pairs; stage 1 samples real and fake pair transforms independently under the same
policy. Generator adversarial updates use the same plane policies.
Only explicit per-plane augmentation policies are accepted in configuration.

### Training diagnostics and continuation

LR records `metrics.jsonl` and TensorBoard scalars for each completed step,
including timestep-specific losses, plane losses, critic scores, unscaled
parameter-gradient norms, optimizer skips and AMP scale. On the regularization
cadence it also measures the global critic's real-input gradients with respect
to both the reconstructed slice and the noisy conditioning slice. A strong
conditioning gradient is a diagnostic signal, not proof of a shortcut; generated
and real conditioning distributions still differ in this 2D-supervised method.
SR records adversarial/consistency terms, per-plane scores and R1/R2 separately.

LR regularization uses each critic's successful update count, including
the intermittent connectivity critic. A shared AMP scaler updates once per LR
iteration. Non-finite losses fail before the corresponding optimizer step;
non-finite FP32 gradients fail, while AMP gradient skips are recorded. EMA only
advances after a successful generator update. No gradient clipping or adaptive
augmentation schedule is enabled without evidence that it is needed.

Anchor supervision and continuity have independent schedules:
`conditioning.anchor.start_step/ramp_steps` default to 0/500, while
`loss.connectivity.start_step/ramp_steps` default to 0/20000.
Measured anchor positions are excluded from parallel fake slices for both critic
and generator updates. Crossing slices remain available to judge their surroundings.

Final-transition volumes generated with visible measured anchors enter a bounded
per-domain replay bank (`conditioning.anchor.bank_capacity`, default 4). Replay
always includes the original measured plane, plus generated planes at a density
controlled by `conditioning.anchor.plane_spacing` (default 16 grid cells).
The measured plane retains pixel supervision; generated planes are coarse targets.
No reference reverse pass is run. Connectivity uses detached replay volumes as
pseudo references, never as measured 3D ground truth. The bank is checkpointed.
Pseudo-anchor intersections are reconciled with the measured planes, but the
connectivity critic and transition loss use the replay before that overwrite.
This keeps pasted-plane jumps out of continuity targets; anchor loss separately
supervises the measured values.

`loss.connectivity.windows_per_plane` defaults to 4 local windows per plane and
orientation. Sampling uses integer plane regions. The first window uses gap 1;
others sample up to `max_slice_gap`. Matched windows share crops and augmentation.
`anchor_neighbor_agreement` compares generated neighbors with the measured plane;
`anchor_neighbor_excess_jump` subtracts the measured plane's in-plane variation.
These diagnostics work without a generated reference and are not connectivity guarantees.

Both stages evaluate area-average critic pyramids down to
`model.critic.pyramid_min_size` (default 16), averaging losses across levels.
`train.structure_every_steps` (default 100, 0 disables) logs phase-specific 6-connected
percolating fractions, straight-path lower bounds, and axis gaps for shared critic
groups. Generated slices are also compared with measurements using directional
2-point correlation and chord-length distribution distances. Chords include runs
censored at crop edges on both sides. These finite-volume diagnostics require no
3D truth, but do not establish target tortuosity or reconstruction accuracy.

Scalar logging values are transferred together at the end of a step. Finite-loss
and finite-gradient checks remain synchronous before optimizer updates.

Real pair-critic `current` inputs come from measured-image forward diffusion;
fake `current` inputs come from the detached generated reverse chain. These
conditional distributions can differ, allowing a critic to distinguish their
source without using `previous`. Source code alone does not establish that this
shortcut dominates training, and extending R1 would not equalize the distributions.
On every `loss.r1_every_steps` successful-generator-update cadence, both stages
record `generator_input_gradient/<group>/<axis>/t<transition>/{previous,current}`.
These are mean per-example L2 norms of each plane's generator adversarial
objective input gradients, including local-head weighting and pyramid averaging,
before plane-group averaging and with the batch-mean factor removed.
This uses the generator-update critic after its update;
the diagnostic `current` leaf remains detached from the preceding reverse chain.
Zero `previous` sensitivity with nonzero `current` sensitivity identifies no local
restoration signal from that critic on the sampled inputs. It does not by itself
prove a distribution shortcut or measure the full denoiser parameter gradient.
Missing values mean no diagnostic was emitted (off cadence or no eligible fake
slices). Losses, R1/R2 targets and
training-pair construction are unchanged. Diagnostics require an extra critic
backward traversal on scheduled updates, but no extra sampling chain or fixed seed.

Two optional advanced settings support controlled comparisons:

```yaml
model:
  diffusion:
    time_embedding: scaled  # default index; scaled uses 1000 * t / num_steps

loss:
  r1_weight: 0.1
  r2_weight: 0.1  # default 0; shares the R1 interval
```

This is a partial override example. In both stages, time embedding is shared by the generator,
pair critics and inference loader; changing it requires a new experiment and its
saved configuration. It does not change the diffusion schedule. R2 regularizes
detached fake inputs during critic updates, with the same local/global weighting
as R1. It also applies to the LR connectivity critic. SR uses these same
timestep-pair losses and R1/R2 regularization. Short comparisons do not establish a better
default. The current presets use `index` time embeddings and enable both R1 and R2; omitted R2 still defaults to zero.

LR still exports `generator.pt` (EMA) and the existing critic weight files. It
additionally saves an atomic `checkpoints/last.pt` containing the online model,
EMA, all optimizers, scaler, step/update counts and anchor-selection state.
The checkpoint records source
image hashes and rejects changed images on resume:

```bash
python run_train_1st.py --resume run/<lr-run>/checkpoints/last.pt --steps 30000 --device cuda
```

`--steps` is the total target. Resume creates a new run and uses saved settings;
`--config` and `--data` cannot override them. LR and SR training do not set a
fixed seed or save/restore RNG state and data order. Resuming retains learned
weights and optimizer progress while drawing fresh random samples.
`train.seed` is rejected. Reproducibility via `seed` is
provided for LR and SR prediction only. The same-step sharing of noise between
anchored and unanchored reference paths is retained for aligned training pairs;
it does not fix randomness across runs.
`initial_weights` remains weight initialization and does not
restore training state. After an interruption inside a step, resume from the last
completed checkpoint; partially updated models are not saved as resumable state.

The default configuration learns 64³ volumes. Train new weights for the current
fractional-occupancy preprocessing and conditioning contract.

### Stage 2: super-resolution

Set the frozen stage-1 model in `config/train/sr.yaml`:

```yaml
source:
  weights: run/<lr-run>/generator.pt  # the run folder is also accepted
```

```bash
python run_train_2nd.py --device cuda
```

Configured relative weight paths resolve from the project root, regardless of
the working directory. `--base-weights` overrides `source.weights` for a new run;
relative CLI paths resolve from the working directory. The selected absolute
weight path and source hashes are saved with the run. A missing or empty source
is rejected before creating a run. Resume uses the source saved in its checkpoint.

The command reads [`config/train/sr.yaml`](config/train/sr.yaml) and the data preset
it selects; `--config` and `--data` can select alternatives. It checks the chosen
crop/LR sizes, phase count and domain IDs against the stage-1 run's saved
`train.yaml` before generating any LR bank. Image folders may differ; their domain
and phase meanings must agree. SR settings are not overwritten by stage-1 data or
augmentation. The selected data preset supplies `hi_res_size`.
`lr_bank.guidance` is independent of `config/gen.yaml`.
For each domain it caches LR volumes from the frozen stage-1 model, then trains
another `Denoiser3D` with the same diffusion equations, timestep-pair critics and
`Trainer` used by stage 1. SR critics remain separate per domain and follow the
selected plane groups. Real sections are resized from original crops to the HR
grid before comparison with generated HR slices.

SR conditions every reverse step on trilinearly enlarged coarse phase fractions,
the realized coarse corruption level, the domain and optional height. It accepts
neither anchors nor volume-fraction requests. Domain CFG retains coarse and height
in both branches. The additional loss compares area-averaged clean HR predictions
with the uncorrupted LR fractions; a tolerance lets boundaries adapt. No measured
HR 3D target is required. Generated LR conditions are not 3D ground truth.

Each block now runs a complete reverse diffusion chain. This costs more than the
previous single-forward SR CNN. Train new SR weights for the diffusion architecture;
legacy SR models are not retained or converted.

Both stages use the same run layout, rooted at the project directory. The optional
top-level `nickname` is appended after the stage:

```yaml
stage: low_res  # sr for stage 2
nickname: experiment_a
data: config/data/default.yaml
```

This produces `run/MMDDHHMM_low_res_experiment_a/` or
`run/MMDDHHMM_sr_experiment_a/`. Omit `nickname`, leave it empty, or use `null` for
`run/MMDDHHMM_low_res/` and `run/MMDDHHMM_sr/`. Surrounding whitespace is trimmed;
path separators and Windows filename characters are rejected. Collisions add a
sequence before the stage, such as `MMDDHHMM_02_sr_experiment_a`. An explicit
`--run-dir` is used exactly as supplied and must not already exist. Existing runs
are not renamed; resumed runs retain their saved nickname.

```text
run/<timestamp>_<stage>[_<nickname>]/
  train.yaml                   # resolved data and training settings
  data_manifest.json           # source images, geometry, hashes and holdouts
  generator.pt                 # EMA inference weights
  critic_<group>.pt             # one weight file per critic group (and SR domain)
  metrics.jsonl                # per-step losses and diagnostics
  tensorboard/                 # the same scalar tags in both stages
  lr_bank/step_*.pt            # SR only: initial bank (step 0) and refreshes
  checkpoints/last.pt          # resumable model, EMA, optimizers and scaler
  checkpoints/step_*/          # archived generator and critic inference weights
```

`train.weights_every_steps` saves latest inference weights and `checkpoints/last.pt`;
`train.archive_every_steps` stores numbered inference snapshots (`null` disables
archives). Both stages save latest weights and resumable state on normal completion.
On interruption they export current weights without replacing the last completed
training checkpoint. Both checkpoint writers use atomic replacement.

LR additionally exports `critic_c.pt`. SR stores immutable banks under `lr_bank/`,
starting with `step_00000000.pt` and adding `step_XXXXXXXX.pt` at each refresh.
Retain the bank referenced by each checkpoint, including references to prior runs
when resuming. Existing run files are not moved.
SR's `generator.pt` embeds its configuration and artifact-kind tag; LR's
`generator.pt` uses the adjacent `train.yaml`. The two model formats remain distinct.
Both inference APIs accept a run folder, an archived step folder, or a generator file.

Both training CLIs delegate setup to `src/train/run/low_res.py` and
`src/train/run/sr.py`. The shared loop in `src/train/run/loop.py` writes resolved settings,
the data manifest, JSONL metrics and TensorBoard events, and manages save schedules
and interruption. SR supplies its bank refresh before each scheduled training step.

`--bank-size` controls samples per domain. `lr_bank.refresh_every_steps` defaults
to 1000 when omitted (the current SR preset uses 500); each refresh replaces one
sample per domain using fresh noise from the same frozen LR weights. Set it to 0
to disable refresh. Source weight and
configuration hashes are checked before refresh. Each refreshed bank has a
separate filename, so an earlier
checkpoint still references its original bank.

SR applies spatially correlated phase replacement to the conditioning input with
`conditioning.coarse_corruption_probability: 0.5` and maximum patch selection
probability `coarse_corruption_strength: 0.2`. The consistency target remains the
uncorrupted generated LR volume. Setting either value to 0 disables corruption.
This supplies corrupted/clean coarse pairs, not measured 3D truth; it cannot
establish correction of systematic LR bias without held-out experiments.
`coarse_corruption_mse` records the actual input perturbation. The SR model also
receives the realized changed phase mass as `corruption_level`; clean inference
always uses level zero. Validate on LR
seeds outside every training bank.

```bash
python run_train_2nd.py --resume run/<sr-run>/checkpoints/last.pt --steps 30000 --device cuda
```

Resume uses saved settings and the bank referenced by the checkpoint, writes a new run,
and treats `--steps` as the total target. Keep the source images unchanged too.
Optimizer progress is restored while data and noise are sampled anew.

Each run saves its resolved data fields and training settings, rather than a
reference to an editable data preset: both stages save `train.yaml` beside their
weights and embed resolved settings in their training checkpoints. Resume
rejects `--config`, `--data` and `--bank-size`; only `--steps` changes the target.
Both stages require explicit data selection and current configuration keys.

## Generate

```python
from src.api import InferenceAPI, PlaneAnchor

api = InferenceAPI("run/my-experiment/generator.pt", device="cuda")
anchor = PlaneAnchor(api.prepare_image(original_crop), axis=0, index=0)

direct = api.generate(seed=0)
anchored = api.generate(anchors=(anchor,), seed=0)
scaled = api.generate(anchors=(anchor,), blocks=(3, 3, 3), seed=0)
```

`PlaneAnchor.image` accepts integer H,W labels or floating C,H,W phase fractions.
`prepare_image` returns C,H,W fractions at LR spacing. With `blocks` or `shape`,
anchor indices and positions refer to the complete output volume, not one block.
Large planes are clipped into every intersecting tile; `position=None` centers
the image within the full output plane. No preliminary anchored base block is
created. This changes the old implicit base-placement behavior.

Anchored inference uses the same conditional denoiser and one evolving diffusion
state as training. `anchor_strength` scales the input mask and its multiscale
features. At `guidance=1` there is one forward per transition at every strength;
nontrivial classifier-free guidance needs at most two. Anchors are learned conditions, not overwritten output
labels. Gaussian spread, temporal correction and coupled-state sampler options
have been removed.

The existing `generate` and `scale-up` paths return LR labels at the model's
trained voxel spacing. Scale-up extends the field of view. Super-resolution
increases the number of voxels representing the same field of view:

```python
from src.api import SuperResolutionAPI

sr = SuperResolutionAPI("run/<sr-run>/generator.pt", device="cuda")
high = sr.super_resolve(direct, seed=0)
# Tile size, overlap and margin are HR voxels; tiles share one diffusion chain.
high = sr.super_resolve(scaled, seed=0, tile_size=128, overlap=16)
```

Fractional scales are supported for one block when all output dimensions are
integral. Multiple tiles require an integer HR/LR ratio and HR shape, tile size,
stride (`tile_size - 2 * overlap`) and margin aligned to coarse voxels. Each tile
reads its coarse region plus one LR voxel of interpolation halo, upsamples it,
then discards the halo. This matches global trilinear interpolation without
allocating the entire enlarged condition on the GPU. SR uses the same global
diffusion sampler as LR: all tiles read one current state and share a latent
per timestep. Clean predictions are blended in a circular slab, then each
voxel's posterior is updated once. Tile context participates in tapered overlap;
only the outer global margin is cropped from the result. This synchronizes the
stochastic trajectory, but finite tile context still requires seam evaluation.

`height_origin` is the source-pixel origin of the supplied global LR volume.
Tile context starts at `(tile_start - margin) * crop_size / hi_res_size` relative
to that origin, so overlapping materials receive identical height coordinates.
The CLI generates the global LR volume once; it does not regenerate LR conditions
for individual HR tiles. At the outer boundary coarse context is replicated.
When supplying a coarse volume that includes additional context below the desired
output, its origin must likewise include that negative context offset.

SR does not accept plane anchors or VF as model conditions; those belong to
stage 1. Its sampler can preserve an existing HR subvolume using `base` and
`base_offset`. Prefer fractional LR `.pt` input over a discretized LR TIFF.

CLI generation saves the HR TIFF, its LR TIFF and a JSON resolution record:

```bash
python scripts/06_check_hr.py --weight run/<sr-run> --no-view
python scripts/06_check_hr.py --weight run/<sr-run> --input run/checks/<result>/lr_probs.pt --no-view
```

Extent growth and conditioning combinations are available through the Python API
and the numbered inspection scripts:

To inspect a completed HR model interactively, run
`python scripts/06_check_hr.py`. It selects the newest SR export directly in
`run/` and uses that export's saved LR source. It saves `lr.tiff`, `lr_probs.pt`,
`hr.tiff`, `comparison.png` (LR / nearest baseline / HR in xy, xz, yz), and
`report.json` under a new `run/checks/<time>_06_check_hr/` directory. The default opens the
comparison figure; `--napari` opens spatially aligned LR/HR 3D layers instead.
Use `--no-view` for saving only. The occupancy comparison is a coarse-consistency
diagnostic, not a reconstruction score against measured 3D truth.

```bash
python scripts/06_check_hr.py
python scripts/06_check_hr.py --weight run/<sr-run> --napari
python scripts/06_check_hr.py --weight run/<sr-run> --input run/predictions/low_lr_probs.pt --no-view
```

| Python API path | Status | Arguments |
|---|---|---|
| LR: none / VF / plane anchors / anchors + VF | Implemented | `vf`, `anchors` |
| LR directly at larger extent, with the same four conditions | Implemented | `shape=(D, H, W)` |
| Existing LR → larger LR, including repeated extensions | Implemented | `base`, `shape`, `base_offset` |
| Existing LR + additional anchors + VF → larger LR | Implemented | Combine the preceding controls |
| Single or extended LR → HR of the same field of view | Implemented | `sr.super_resolve(low, tile_size=...)` |
| Existing HR → larger field of view, preserving old HR | Implemented | `extend_hr(lr, sr, base, shape)` |
| LR extension → HR → further HR extension, repeatedly | Implemented | Reuse the prior HR and its saved LR fractions |
| SR alone synthesizes a larger field without an expanded coarse volume | Not supported | Extend LR first, then refine |
| Plane anchors / VF directly into the SR network | Not supported | Apply them during LR generation |

LR `base` accepts arbitrary D,H,W label volumes or C,D,H,W fractions. By default
it guides the reverse chain and may change. `preserve_base=True`
fixes the known region throughout sampling; contradictory overlapping anchors
are rejected. Additional anchors are still learned conditions, not exact output
constraints. Fully known tiles skip denoiser forwards; boundary tiles still run
to provide context. Both paths retain the synchronized global diffusion chain.

HR extension uses `from src.api import extend_hr`. It returns `(lr_fractions, hr)`:
`extend_hr(lr, sr, old_hr, final_hr_shape, low=old_lr, base_offset=(0, 0, 0))`.
The offset and output shape are HR voxels; extra anchors use the final LR grid.
Extent growth requires an integer SR ratio and aligned HR sizes/offsets. Omit
`low` to derive fractional coarse conditions by area-averaging the HR phase
channels. Supplying original LR fractions avoids that approximation. Existing
HR labels remain exact; fractional inputs pass through the sampler's fp16 state.
Height-conditioned extension also requires `base_height_origin` consistent with
the offset and final `height_origin` (both in source-image pixels).

```bash
# Exercise the matrix with tiny untrained networks, or supply trained weights.
python scripts/experiments/prediction_combinations.py --smoke
python scripts/experiments/prediction_combinations.py --lr-weights run/<lr-run> --sr-weights run/<sr-run> --device cuda --anchor-image data/crop.png
```

The checker saves each stage as TIFF, LR fractions as `.pt`, center-slice PNGs,
and a `report.json` with shapes, phase fractions and preserved-label checks.
Without `--anchor-image` it uses a synthetic phase pattern. The smoke option
checks execution and preservation, not trained generation quality.

Changing `hi_res_size` in a data preset does not convert an existing trained SR model
to a different scale. Train weights for the intended scale and original field of
view. Input TIFFs must contain integer phase labels with the same channel meanings
and LR voxel spacing as training. Physical units require the source pixel spacing;
without it, generated sizes are reported in voxels/source pixels.

## Code layout

```text
frontend/            Vue UI; npm run dev/build/test
backend/
  run.py             HTTP server entry point
  config.yaml        request, output and download limits
  src/               HTTP schemas, middleware, routes and responses
simul/
  run.py             simulation entry point
  config.yaml        default synthetic geometry and export settings
  config_height.yaml height-varying geometry preset
  src/               synthetic geometry and export
src/
  config/            YAML I/O, defaults, schemas and resolved settings
  build/             model, data-loader and trainer assembly
  model/             neural networks and diffusion equations
  data/              image sources, datasets, loaders, LR banks and augmentation
  prepare/           shared phase conversion and resolution transforms
  train/             stage-1 and SR training, losses and EMA
  predict/           generation, volume extension and SR inference
  evaluate/          morphology, connectivity and transport measurements
  storage.py         volume and model artifact I/O
  api.py             stable public Python exports
```

Model code follows the shared `D:/code/guide.md`, linked by `AGENTS.md`.
This project keeps the web UI, HTTP server and simulation as root-level packages.
`backend/run.py` and `simul/run.py` own their command-line entry points.
The frontend keeps HTTP encoding in `api.js` and request state in `use-generation.js`.
Test and lint settings are kept in `pyproject.toml`.
Model factories live in `build/model.py`; loading the stage-1 inference generator
lives in `build/predict.py`. Training assembly lives in `build/trainer.py` and
`build/sr.py`. Import these owner modules directly: `build/__init__.py` does not
re-export them, so importing the public inference API does not load training.

`train/trainer.py` owns both stages' diffusion updates. `train/batch.py` defines
step inputs and results; `train/metrics.py` materializes and records diagnostics.
`train/run/loop.py` owns the shared loop and save schedule; `low_res.py` and `sr.py`
in the same folder handle stage setup and resume. `train/run/bank.py` creates and
refreshes immutable LR bank snapshots. `train/sr.py` owns coarse corruption,
SR training state and inference exports; `config/` owns configuration validation.
The root training commands parse CLI arguments.
Step counts and save intervals must be positive integers; omit
`archive_every_steps` (or set it to null) to disable archival checkpoints.

`train/loss/` contains loss calculations, including SR coarse
consistency in `sr.py`, transition consistency in `connectivity.py` and phase-fraction
loss in `volume_fraction.py`. `data/slice.py` samples sections, aligned diffusion
pairs and anchor triplets; `evaluate/` owns measurement calculations. `data/dataset.py`
contains the phase-fraction dataset, `data/loader.py` contains its batch stream,
and `data/bank.py` reads and validates fractional LR banks. `data/source.py`
discovers source images and infers height extents.
`plane.py` owns plane names, numeric axes and row/column directions.

`predict/tiling/sampler.py` runs tiled diffusion and continuation. Its sibling
modules `layout.py`, `state.py` and `fusion.py` own tile geometry, volume buffers
and overlap blending. `predict/sr/` contains SR inference, extension and memory
budgets; `predict/random.py` isolates seeded inference RNG state. Evaluation files name their
measurement: `fid.py`, `connectivity.py`, `seam.py` and `tortuosity.py`.
Import metrics from their owning modules. FID and KID share image conversion in
`evaluate/image.py`: binary masks use bool, floating-point images use [0,1],
and uint8 images use [0,255], regardless of batch contents.
Tests follow these source responsibilities under `tests/data/`, `tests/model/`,
`tests/prepare/`, `tests/predict/`, `tests/backend/`, `tests/train/` and
`tests/evaluate/`; cross-module tests stay at the test root. Internal imports use
the owning modules directly. The public
`src.api` exports and model state-dictionary formats remain available.
Existing paper assets and run files retain their paths. `PAPER.md` records prior
128³ experiments, not results for this new LR/SR configuration.
Generate synthetic data with `python -m simul.run --config simul/config.yaml`.

For a known 3D reference with a thickness-dependent particle-size distribution:

First adapt the training preset to the z-thickness constraints described above:
disable quarter-turn rotations in `xz`/`yz`, allow only x flips in `xz` and y
flips in `yz`, and choose critic groups appropriate to the anisotropic data.

```bash
python -m simul.run --config simul/config_height.yaml
python run_train_1st.py --data config/data/simul_height.yaml --device cuda
```

Enable `conditioning.height_enabled: true` in the chosen training preset for
the height-conditioned experiment. `geometry.radius_gradient: [0.6, 1.4]`
multiplies both particle radii linearly from z=0 to the last exported z plane;
the existing elongation remains along z. Padding extends the endpoint radii.
The output includes categorical 3D TIFF references, correctly oriented xy/xz/yz
sections, and `simulation.yaml` recording the geometry. The data preset crops
64 source pixels from full 128-pixel-thick side images, so crop origins provide
height supervision. Reserve independent simulated volumes for evaluation;
neighboring sections of a training volume are not held-out specimens.

`src.evaluate.compute_kid` complements FID with the unbiased polynomial MMD²
estimator. It requires at least two samples per set and defaults to 100 subsets
of `min(50, n_real, n_generated)` images, so 64-image comparisons are supported.
Negative finite-sample values are retained. The structure evaluation script
records KID mean, subset standard deviation and sampling settings alongside
FID, using the same 192-dimensional features. The subset deviation is not a
confidence interval across independent volumes; correlated sections and
overlapping crops still limit comparisons. Existing recorded paper results
are unchanged and contain no retroactively inferred KID scores.

## Verification

```bash
python -m pytest tests/test_sr.py tests/test_sr_pipeline.py -q
python -m pytest tests -q
python -m ruff check src scripts tests run_train_1st.py run_train_2nd.py backend simul
```

A bounded GPU check compares index/R1, scaled-time/R1 and index/R1+R2, verifies
LR continuation, then runs generated-LR-bank SR and exports held-out-seed volumes:

```bash
python scripts/experiments/training_stability.py --image data/sample.png --steps 100 --sr-steps 100 --device cuda
```

The script uses 128px source crops, 64³ LR and 128³ HR, reduced networks, a short
anchor ramp and faster EMA for the smoke experiment. It splits one image into
disjoint train/validation regions, with isotropic reuse across planes. This is
not an independent-specimen validation or a battery thickness experiment. It
saves full experimental settings, logs, validation TIFFs and `report.json` under
`run/`. Production recipes are not replaced by a winner from this short test.

For scientific evaluation, compare against nearest-neighbor and phase-channel
interpolation baselines on held-out LR seeds and HR image regions. Check coarse
agreement, axis-wise HR section statistics, phase fractions, percolating fractions,
tortuosity and tile seams. Report distances and surface measurements at matching
physical scales. A training smoke test establishes execution, not fine-structure
accuracy or a unique reconstruction.

## Web interface

Build the Vue frontend once after cloning or changing files under `frontend/`:

```bash
cd frontend
npm ci
npm run build
cd ..
```

The generated `frontend/dist/` directory is intentionally not committed. Start the
API after the build completes:

```bash
python -m backend.run --weight run/my-experiment/generator.pt --device cuda
```

Edit `backend/config.yaml` to adjust dimension, voxel, block, anchor, JSON-body
and concurrent-download limits, then restart the server. `--config` selects another
file; `--max-inflight-downloads` overrides that setting for one run. Configuration
keys are strict and all limits must be positive integers. The 256-phase ceiling
is fixed by uint8 output and is not configurable.

Open <http://127.0.0.1:8000/> to crop an input section, generate a 3D volume,
inspect its phases, and reveal its continuation along axis 0.
PNG label sections are decoded from their raw values: indexed PNG palette
indices, grayscale PNG samples, or RGB PNGs whose three channels are identical.
No luminance conversion or palette-color mapping is applied. The selected original
crop is prepared by the server using the model's saved preprocessing contract:
phase-channel original-crop→LR resizing with fractional occupancy retained.
`/prepare` returns a C,H,W array accepted directly by `/generate`.
The web endpoint continues to generate stage-1 volumes; use the SR Python API or
CLI above for the separate super-resolution pass.

Tiled LR generation stores two fp16 diffusion states and blends predictions in
a circular slab only one tile deep. `storage="auto"` checks live CUDA free memory
with a workspace reserve and selects GPU or CPU state storage. Explicit
`storage="cpu"` / `"cuda"` is supported by both `generate()` and tiled
`generate_probs(shape=..., storage=...)`. Probability conversion also uses
bounded chunks; the final probability tensor is returned on CPU.
`src.predict.memory.estimate_memory(shape, num_phases, tile_size=..., margin=...)`
reports state, slab, scratch and output budgets. Network workspace estimates
are conservative approximations, not an OOM guarantee. Both states and the
returned output still scale with volume size in RAM; disk-backed states are
not implemented.

Single-volume label generation selects `argmax` on the model device and transfers
only uint8 labels. Probability requests retain the normalized fractional output
and have a separate output-memory budget. Phase-channel conversion writes directly
into float32 storage without an expanded int64 one-hot intermediate.

SR checks its own RAM/VRAM budget before coarse conversion or HR allocation:
`src.predict.sr.memory.estimate_sr_memory` includes LR phase conversion, two global
CPU fp16 states, a circular accumulation slab, interpolation halo, and
margin-expanded model tiles. State and output storage still scale with HR volume
size; the estimate is not a guarantee against OOM.
`output_kind="labels"` also budgets the final uint8 buffer and bounded argmax
workspace. Single-block SR selects labels on the model device; tiled SR first
completes the shared diffusion chain, then selects labels in depth slabs (up to 1,048,576 voxels,
or one plane when a plane is larger). LR/SR probability conversion reuses owned
float32 model outputs.

The default server configuration bounds axes to 1024, total output voxels to 512³, block counts to
64 per axis / 4096 total, and anchors to 32. The actual tile count, including
overlap and margins, also cannot exceed 4096. Resolved block dimensions are
checked against the same volume limits before allocation. JSON request bodies
are limited to 16 MiB, including chunked uploads. Available CPU/GPU memory is
checked before generation; memory-budget failures and PyTorch CUDA OOM return
HTTP 413. Invalid dimensions or counts return HTTP 422.

Concurrent generation requests return HTTP 503 immediately with `Retry-After: 1`.
Separately, `create_app(max_inflight_downloads=2)` bounds retained results per
server process, including generation and active transfers. Configure the limit
with `python -m backend.run --max-inflight-downloads`; full capacity returns the same 503
before generation. Slots are returned after completion, disconnect, cancellation,
or preparation failure. Slow downloads retain their slots but release the GPU
generation lock so another request can run when a result slot is available.
`include_metrics` defaults to `false`; set it to `true` to calculate porosity and
tortuosity and include their response headers. The GUI explicitly requests these
metrics. GPU metrics run under the same generation lock to avoid resource races.
Raw labels are streamed as bounded 64 KiB views; TIFF responses use a temporary
file, closed and deleted after completion or cancellation. File-writing failures
are handled before response headers: disk-full errors return HTTP 507, other I/O
errors return HTTP 500, and memory failures return HTTP 413.

Seeded inference changes and restores only the CPU and selected CUDA device RNG;
unrelated GPU RNG states are left untouched, including when inference raises.

## Citation

```bibtex
@software{phykn2026anchorconditioneddiffusion,
  author = {phykn},
  title = {Anchor-Conditioned Diffusion for Scalable 3D Microstructure Synthesis},
  year = {2026},
  url = {https://github.com/phykn/diffusion-gan3d}
}
```

Generated LR → SR inference keeps fractional channels throughout. The CLI saves
`<output>_lr_probs.pt` for lossless reuse with `--input`, plus `<output>_lr.tiff`
for label-based inspection. Supplying a label TIFF intentionally supplies one-hot
coarse data instead. `InferenceAPI.generate_probs()` and SR's `predict_probs()`
provide the fractional Python path.


### Height-dependent profiles from 2D images

Only 2D images are used as training observations. In the LR preset, enable:

```yaml
conditioning:
  height_enabled: true
  spatial_profile:
    enabled: true
    num_bins: 16
    critic_enabled: false  # optional conditional-critic comparison

loss:
  spatial_profile_weight: 1.0
  spatial_profile_gradient_weight: 0.0
```

The data preset must specify `thickness_axis: z`. Preserve z in augmentation:
`xz.flip_axes: [x]`, `yz.flip_axes: [y]`, and `rotate_90: false` for both.
Each crop supplies its own profile. LR uses a spatial adapter, masked local loss,
joint profile/VF CFG dropout and replay with the original conditions. No 3D truth
is required. `profile/soft_mae` and `profile/label_mae` use fractions (0.01 means
one percentage point); passing software tests does not establish generation quality.

```python
profile = {
    "axis": "z",
    "points": [[0.0, [0.1, 0.9]], [1.0, [0.5, 0.5]]],
    "interpolation": "linear",
}
volume = api.generate(
    domain=0,
    seed=7,
    height_origin=256,
    height_extent=1024,
    vf_profile=profile,
)
```

Knots span the **full reference image** from 0 to 1. `linear` and `constant`
(left-interval values) are integrated over output cells. With crop=128 and LR=64,
a 64-voxel output at origin 256 spans 25–37.5% and its phase-0 target mean is 22.5%.
A supplied `vf` must equal the profile average over the final output interval.
Tiles use shared source coordinates and tile-local VF. Full-strength anchors and
preserved base volumes are checked for incompatible phase fractions.
`generate_probs` and HTTP accept the same dictionary. CLI uses
`--vf-profile profile.json --height-origin 256 --height-extent 1024`.
SR receives generated LR fractions plus the same origin/extent; direct profile/VF
input remains LR-only. SR banks record the crop source, profile and per-image height.

Explicit holdouts can be specified in the data preset:

```yaml
split:
  validation_files: [data/material/independent_sample.png]
  validation_regions:
    data/material/side.png: [0, 0, 128, 512]  # top, left, height, width in source pixels
```

Paths must be images in the configured plane folders. Training excludes validation
files and every crop intersecting a reserved rectangle; each observed plane must
retain training images. Full source heights are preserved even for region holdouts.
`data_manifest.json` records source sizes, hashes and the explicit split. Region
holdouts from one image test spatial interpolation, not independent-specimen
accuracy. No split is invented automatically. Evaluate held-out 2D statistics
separately; generated 3D connectivity is a diagnostic, not measured 3D accuracy.
