# Run and inspect

Run from the project root. `--weight` accepts a run directory, its `generator.pt`,
or an EMA export under `checkpoints/`. Results go to a new folder under `checks/`;
add `--no-view` to skip the viewer in `01–06`.

| Script | Purpose | Weights |
| --- | --- | --- |
| [01_dataset.py](check/01_dataset.py) | Preview the same training crops at the LR and SR grids | None (`--sample`) or a run's data (`--weight`) |
| [02_generated.py](check/02_generated.py) | Generate an LR volume | LR |
| [03_anchor.py](check/03_anchor.py) | Compare anchored and unanchored samples | LR |
| [04_scale_up.py](check/04_scale_up.py) | Tiled generation and seams | LR |
| [05_continuation.py](check/05_continuation.py) | Grow a volume from a boundary section | LR |
| [06_hr.py](check/06_hr.py) | Compare LR, upsampling, and SR | SR |
| [07_tilt.py](check/07_tilt.py) | Signed particle tilt balance vs real volumes | LR (optional) |

```bash
python scripts/check/04_scale_up.py --weight "run/my-run-lr" --blocks 2 2 2 --no-view
python scripts/check/05_continuation.py --weight "run/my-run-lr" --anchor data/sample.png --axis 0
```

- `--blocks D H W` sets the `(z, y, x)` size to `patch + (n - 1) * (patch - 2 * overlap)`,
  where `patch` is the LR `lo_res_size` (default 64); the script prints the actual tile grid. `--axis 0/1/2` selects `xy/xz/yz`.
- `--seed` and `--domain` work in `01–07`, `--device` in `02–07`, and `--napari` in `02–06`;
  `--help` lists the rest.
- `--height-origin` and `--height-extent` (`02–07`) are in source pixels along the xz/yz row direction.
- `02 --vf-profile scripts/simul/config_anode.yaml --guidance 1.5` fixes the depth fractions to
  that simulation's targets (a profile YAML/JSON also works). Without it, a
  profile-conditioned model follows its training depth profile, since height alone
  flattens a strong gradient.
- `01` saves `crops.png`; `02–05` save `volume.tiff/png/json`, and `05` also
  `volume_continuation.png`; `06` saves `lr.tiff`, `hr.tiff`, `lr_probs.pt` (for fractional
  LR input), `comparison.png`, and `report.json`; `07` saves `report.json`.
- `07` counts particles tilted each way (`dz/dy`, `dz/dx`, `dy/dx`); generated volumes
  should match `--real` (use `--real-stride 2` for 128-pixel crops at 64 voxels); for a
  height-conditioned model, only at the real depths the generated volumes span from
  `--height-origin`. Training logs the same balance as `structure/phase_N/tilt_balance_*`
  (pore phase 0 excluded);
  percolation is logged per transport axis as `structure/phase_N/percolation_z|y|x`, and
  section errors against real crops as `structure/<plane>/two_point_mae|chord_tv|cluster_error`.
- Guidance in `02–05` and `07`, the overlap of `04`, and the anchor strength of `03–05`
  (posterior-sampling step scale, 0 = off) default to [config/gen.yaml](../config/gen.yaml);
  `04` keeps its default overlap below half a patch. In `06`, `--lr-guidance` uses that
  default, `--guidance` is the SR guidance (default 1.0), and overlap is scale-aligned.
  Anchor demos show controlled generation, not 3D accuracy.

## Training options

On macOS, `scripts/commands/train_default.command` (and `train_anode.command` for the simulated
anode) runs the README training steps on double-click in Finder; the `_1st`/`_2nd` variants split LR
and SR. `STEPS=200 SR_STEPS=200 scripts/commands/train_default.command` makes a short trial.
`--resume` continues one interrupted stage ([Save and resume](#save-and-resume)).
During the first `burn_in.steps` steps (LR 5000, SR 2500) a cosine curve moves the
learning rates to `burn_in.lr_scale` (0.25) of `optim.*_lr`, Adam β2 from
`optim.adam_betas` to `burn_in.beta2` (0.99), the R1/R2 weights to
`burn_in.regularization_scale` (0.1; the default sample keeps 0.5) of
`loss.r1_weight`/`r2_weight`, and the EMA half-life from zero to that of
`optim.ema_decay`, as in R3GAN. TensorBoard logs them under `burn_in/*`.
The device is CUDA, then MPS, then CPU unless `--device` is given; mixed precision runs
in training on CUDA and MPS (on MPS for the denoiser only; its critics stay fp32) unless
`train.mixed_precision: false`, and in LR and SR prediction on CUDA and MPS. Gradient
checkpointing is off; when memory is short, `model.gradient_checkpointing: true` under
`tune.lr` or `tune.sr` costs about 10% more step time for 15–30% less peak memory (Apple
MPS).

### Sample fields

A sample is one file in [config/samples/](../config/samples/); copy
[default.yaml](../config/samples/default.yaml) to add one.

- `images` maps `xy`, `xz`, and `yz` to folders of 2D uint8 label images (`.png`, `.tif`,
  or `.tiff` directly inside) with one shared pixel size; in `xz` and `yz` images, z runs
  down the rows. The default sample reads `data/`. Labels are phases `0..K-1`,
  and phase 0 is pore. `K` is the largest label plus one, and every smaller label must
  appear, unless `num_phases` is set.
- `resolution` sets the data size: each training crop is `crop_size` source pixels on
  an edge (enough to hold the structure), resized to `lo_res_size` voxels for LR and to
  `lo_res_size × sr_scale` voxels for SR (`sr_scale` is an integer of at least 2). Set
  `sr_scale` to `crop_size / lo_res_size` to recover the source resolution; a larger one
  trains SR on upsampled images. Training prints how many source pixels one LR and one
  HR voxel span. Sizes divisible by 8 suit the U-Net, which halves three times.
- `structure` says what the material looks like:
  - `isotropic: false` judges `xy` sections apart from `xz`/`yz` sections.
  - `depth_gradient: true` conditions both stages on height and, in LR, on the phase
    fractions at each height.
  - `mirror_symmetric: true` makes each volume's phase pairs along mirrored diagonals
    agree in LR (`loss.mirror_balance`), so the generator cannot shear the volume to keep two
    section planes realistic at the cost of the third. `false` turns that off, flips and
    rotates sections less often (probability 0.2 instead of 0.5) so a tilt stays visible,
    and never flips SR's LR input volumes (`lr_bank.flips`), which would reverse their tilt.

  When z differs from x and y (either of the first two flags), or no `xy` images are
  judged with the side planes, training never rotates `xz`/`yz` sections, which would
  swap z with x or y; with a depth gradient it also never flips z.
- `tune.lr` and `tune.sr` change single settings of that stage.

Settings apply in order, later winning: code defaults, [config/base/](../config/base/)`<lr|sr>.yaml`,
`structure`, then `tune`. `python train.py default --show` prints the YAML of both
stages without the code defaults.

## Web backend

The web backend (`app/backend/run.py`) serves `GET /api/health`, `POST /api/prepare`, and
`POST /api/generate`; HTTP limits are in [app/backend/config.yaml](../app/backend/config.yaml),
and `--help` lists the host, port, and device options.
Viewer, porosity, and tortuosity are on the HR volume, and one block is
`lo_res_size × sr_scale` voxels per edge.

## Save and resume

- `generator.pt` in the run directory is the inference export, refreshed every
  `weights_every_steps` and at the last step. It is the best scored step so far, not the
  latest one ([Best step](#best-step)).
- Every `archive_every_steps` and at the last step, `checkpoints/` gets `step_N.pt` (full
  training state), `generator_N.pt` (EMA weights), and a `step_N.complete` marker that
  resume requires.
- The first Ctrl+C stops after the current step and saves both; a second stops at once
  without saving.
- Metrics and diagnostics go to `tensorboard/` (`tensorboard --logdir run`); step scores
  also go to `selection/scores.yaml` ([Best step](#best-step)).

```bash
python train.py --resume "run/my-run-lr" --steps 30000 --run-dir "run/my-run-2"
```

Resume continues the stage the checkpoint belongs to with its saved settings, so it takes
no sample, `--sr-steps`, or `--bank-size`; `--steps` is that stage's total, including
completed steps. It writes a new run directory (here `run/my-run-2-lr`); an LR resume
needs `--run-dir`. After an LR resume, `python train.py <sample> --run-dir "run/my-run-2"`
reuses it and trains SR. Pass `--path-map OLD NEW` (repeatable) when data or runs have moved.

- A finished stage holds a `.done` file. While `NAME-lr` or `NAME-sr` exists unfinished,
  `train.py --run-dir NAME` stops and says how to resume or delete it.
- An SR run checks the LR run's `generator.pt` and `train.yaml` against hashes it recorded,
  and an SR resume also reads the resumed run's latest `lr_bank/` snapshot and
  `selection/inputs.pt`; keep them unchanged.
- `06` and the web app load the LR run from the path the SR run recorded; pass
  `--lr-weight` when it has moved.

### Best step

Training improves and then may degrade, for example when the critic memorizes few real
images. At each `weights_every_steps` save and the last step, a run scores its EMA
weights on `selection.samples` volumes against 64 fixed real crops per plane:

- **LR** generates unanchored volumes at guidance 1.0, from fixed seeds whatever
  [config/gen.yaml](../config/gen.yaml) says. Height-conditioned runs take each
  volume's depth from an `xz`/`yz` crop.
- **SR** refines fixed LR volumes, the first ones of the initial bank, with their bank
  heights; bank refreshes do not change them. The real crops are HR sections.

Only 2D sections of the material exist, so the score compares sections:

- **Two-point** and **chord-length** errors, as in `structure/<plane>/*`.
- **Cluster error**: how far each phase's 4-connected clusters per section pixel and
  mean cluster area are from the real crops, relative to their mean. Two-point and chord
  statistics can match while pores merge into large blobs and the solid falls into
  fragments; clusters show what sections reveal of 3D connectivity.

Each error is divided by the same error between two halves of the real crops, so all
three use the sampling noise of the real data as their unit; **structure** is their
mean over planes. `generator.pt` is the step with the lowest structure. Real crops are
labelled by their largest phase; exact ties left by area downsampling are broken at
random, so they do not favor phase 0. Each plane uses
128 generated sections spread evenly over the volumes, so the statistics cost grows with
the section area, not the volume size.

SR also records **lr_mse**, the gap between its downsampled output and the LR input, as
in training's `lr_mse`. It does not affect the choice: SR learns to repair corrupted LR
input, so matching it exactly is not the goal.

- `selection/scores.yaml` lists every scored step and the best step; TensorBoard has
  `selection/*`.
- `selection/generator_N.pt` keeps the best step's weights. Archive checkpoints carry
  them too, so a resume from any checkpoint writes them into the new run. SR runs keep
  their LR inputs in `selection/inputs.pt`; an SR resume reads it from the resumed run,
  so keep it.
- One LR evaluation of 8 samples takes about 4 s per domain at 64³ on Apple MPS; most of
  it is generation, which grows with volume size like a training step does. Set
  `selection.enabled: false` to export the latest step instead.

## Synthetic data

`python scripts/simul/run.py --config <file>` writes 3D TIFFs (`volumes/`), per-axis slice
PNGs (`slices/0..2` for `xy`, `xz`, `yz`), and `simulation.yaml` (targets and measured
fractions) to the config's `output.data_dir`, under `data/generated/` in the shipped
configs. Its `volumes/` and `slices/0..2` must be empty or absent. A sample trains on the
slices by pointing `images` at `<data_dir>/slices/0`, `1`, and `2`, as
[simul_anode.yaml](../config/samples/simul_anode.yaml) does.

- Default `kind: particles` ([geometry.py](simul/src/geometry.py)): elongated phase 2
  and spherical phase 1 particles. y and x wrap periodically, and z does too unless
  `radius_gradient` varies with height.
- `kind: anode` ([anode.py](simul/src/anode.py), [config_anode.yaml](simul/config_anode.yaml)):
  tilted graphite flakes (1) and silicon spheres (2) in pores (0). A `*_vf` list sets
  fractions at evenly spaced heights, enforced per slice. Order bias alone cannot
  concentrate silicon once graphite fills the pores.

The anode sample is not isotropic, has a depth gradient, and is not mirror symmetric; its
settings follow from those flags ([Sample fields](#sample-fields)).

```bash
python scripts/simul/run.py --config scripts/simul/config_anode.yaml
python scripts/check/01_dataset.py --sample simul_anode --no-view
python train.py simul_anode --run-dir "run/anode"
python scripts/check/02_generated.py --weight "run/anode-lr" --no-view
python scripts/check/06_hr.py --weight "run/anode-sr" --no-view
```

## Experiments

[scripts/experiments/](experiments/) holds bounded ablations, not quality scores.
`critic_inputs.py` trains LR runs from the same seeds with the critic seeing section pairs
or single sections, into `--run-dir run/NAME`. `training_stability.py` runs short LR
ablations and an LR → SR run with held-out-region checks, under `run/` by default.

[Back to README](../README.md)
