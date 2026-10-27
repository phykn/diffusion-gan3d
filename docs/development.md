# Development

## Where to make a change

| Change | Start here |
| --- | --- |
| Samples, input images, or resolutions | [config/samples/](../config/samples/), [sample settings](../src/config/sample.py), [RealDataset](../src/data/dataset.py) |
| Network architecture | [src/model/](../src/model/), [model assembly](../src/build/model.py) |
| Training step or objective | [Trainer](../src/train/trainer.py), [losses](../src/train/loss/) |
| Condition sampling or CFG dropout | [ConditionSampler](../src/train/conditioning.py), [step inputs](../src/train/batch.py) |
| Anchors at inference | [anchor encoding and AnchorGuidance](../src/predict/anchor.py) |
| Plane pairs | [section sampling](../src/data/slice.py) |
| Metrics and TensorBoard tags | [metrics](../src/train/metrics.py), [structure metrics](../src/evaluate/structure.py) |
| Best step export | [CheckpointSelection](../src/train/selection.py), [run loop](../src/train/run/loop.py) |
| Section augmentation | [auto plane policies](../src/config/train.py), [CriticAugment](../src/data/augment.py) |
| SR bank or height coordinates | [bank validation](../src/data/bank.py), [SR assembly](../src/build/sr.py) |
| Checkpoints or resume | [run lifecycle](../src/train/run/), [training state](../src/train/state.py) |
| Generation, tiling, or SR | [public API](../src/api.py), [src/predict/](../src/predict/) |
| Web behavior | [backend app](../backend/src/app.py), [frontend](../frontend/src/) |
| Metrics or synthetic data | [src/evaluate/](../src/evaluate/), [simul/](../simul/) |

## How the pieces fit

- Training: `run_train.py` → `src/config/sample.py` builds each stage's settings from
  `config/base/` and the sample → `src/train/run/` → `src/build/` assembles data,
  models, and the trainer.
- `Trainer.step` owns optimizer and EMA updates; it first sets that step's learning
  rates, β2, R1/R2 scale and EMA decay from [BurnIn](../src/train/burn_in.py).
  `Trainer.settings` holds runtime and loss settings built once in
  `src/build/trainer.py`; read them there instead of copying.
- `Trainer.conditions` (`ConditionSampler`) samples real batches, vf/profile targets,
  CFG dropout (`conditioning.cfg_dropout_probability`), and SR coarse inputs. Training
  never sees anchors: the denoiser stays unconditional on known planes.
- `src/train/loss/denoiser.py` owns the generator objective; `src/train/loss/gan.py`
  holds the adversarial losses and R1/R2 penalties shared by every critic. Plane critics
  use the relativistic pairing loss (RpGAN): each generated section is scored against a
  paired real section, so a generator that covers only one mode of the data, such as one
  particle tilt, is still penalized. `Trainer.step` samples real section pairs once and
  both updates use them.
- Phase-pair losses each have a `loss.<name>` section. `real_transition` (LR only)
  matches pairs at in-plane gaps of the final denoised volume (transition 0) to the real
  sections; `group_statistics` keeps the pairs along the normals of planes one critic
  group judges within `tolerance` of each other; `mirror_balance` keeps mirrored
  diagonals equal.
- LR and SR runs score their EMA weights at every weights save with
  [CheckpointSelection](../src/train/selection.py) (`SRCheckpointSelection` for SR)
  before the archive checkpoint, so a resume keeps that score; `save_weights` exports
  the best step as `generator.pt`. Statistics use a fixed number of sections per plane
  (`SECTIONS`), so their cost grows with section area, not volume size.
- With `augmentation.auto_planes`, `auto_plane_policies` in `src/config/train.py` picks
  each plane's flips and rotations. z stays fixed (no `xz`/`yz` rotation) when height is
  enabled or no critic group judges `xy` with a side plane; z flips are off with height.
  SR bank volumes swap or flip z only where those policies do. Set `auto_planes: false`
  to list `augmentation.planes` yourself.
- Generation: [TileSampler](../src/predict/tiling/sampler.py) owns state, noise, diffusion,
  fusion, and output. LR and SR tiling only provide their own conditions. Memory
  estimates in `src/predict/memory.py` and `src/predict/sr/memory.py` run before sampling.
- Anchors at inference never enter the denoiser. [AnchorGuidance](../src/predict/anchor.py)
  runs diffusion posterior sampling: each step moves the noisy state down the gradient of
  the clean prediction's error on the anchor voxels, so neighbors follow the anchor
  instead of meeting a pasted plane at a seam. `anchor_strength` scales that step per
  anchored voxel (0 turns it off); tiles blend their steps like their predictions.

## Contracts to preserve

- Volumes use `(z, y, x)`. Crop and height coordinates are source pixels; LR/HR grids
  are voxels.
- `config/` holds presets, `src/config/` validates them, `src/build/` builds objects.
  Relative YAML paths resolve from the project root.
- LR inference reads the run's `train.yaml` and `config/gen.yaml`; SR reads settings
  inside its export; resume reads checkpoint settings.
- `src/train/state.py` owns training checkpoint formats and the resume config check.
  `src/train/run/resume.py` loads a checkpoint and applies `--path-map` through
  `src/train/relocate.py`; the resumed run's `train.yaml` is the checkpoint's resolved
  config. `src/train/run/source.py` checks the frozen LR source.
- Folders: simulated data in `data/generated/<name>/`, inspection results in `checks/`,
  trained runs only in `run/`.

## Synthetic data

`python simul/run.py --config <file>` writes 3D TIFFs, per-axis slice PNGs
(`slices/0..2` for `xy`, `xz`, `yz`), and `simulation.yaml` (targets and measured
fractions). The output folder must be empty.

- Default `kind: particles` ([geometry.py](../simul/src/geometry.py)): elongated phase 2
  and spherical phase 1 particles. y and x wrap periodically, and z does too unless
  `radius_gradient` varies with height.
- `kind: anode` ([anode.py](../simul/src/anode.py), [config_anode.yaml](../simul/config_anode.yaml)):
  tilted graphite flakes (1) and silicon spheres (2) in pores (0). A `*_vf` list sets
  fractions at evenly spaced heights, enforced per slice. Order bias alone cannot
  concentrate silicon once graphite fills the pores.

The anode sample is not isotropic, has a depth gradient, and is not mirror symmetric, so
both stages use height conditioning and judge `xy` apart from `xz`/`yz`; LR adds the
per-height profile, sections flip at probability 0.2, and z is never flipped or rotated.

```bash
python simul/run.py --config simul/config_anode.yaml
python scripts/01_check_dataset.py --sample simul_anode --no-view
python run_train.py simul_anode --run-dir "run/anode"
python scripts/02_check_generated.py --weight "run/anode-lr" --no-view
python scripts/06_check_hr.py --weight "run/anode-sr" --no-view
```

## Tests and lint

```bash
python -m pip install pytest ruff httpx
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Frontend, from `frontend/`: `npm ci`, `npm test`, `npm run build`. CUDA and MPS tests
skip without that hardware, so CPU passes do not cover AMP. For algorithm changes, also
compare samples at fixed seeds.

[Back to README](../README.md)
