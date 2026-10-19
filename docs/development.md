# Development

## Where to make a change

| Change | Start here |
| --- | --- |
| Input images or resolutions | [config/data/](../config/data/), [RealDataset](../src/data/dataset.py) |
| Network architecture | [src/model/](../src/model/), [model assembly](../src/build/model.py) |
| Training step or objective | [Trainer](../src/train/trainer.py), [losses](../src/train/loss/) |
| Anchors, replay, or condition sampling | [ConditionSampler](../src/train/conditioning.py), [step inputs](../src/train/batch.py) |
| Plane pairs, triplets, or seam bands | [section sampling](../src/data/slice.py) |
| Metrics and TensorBoard tags | [metrics](../src/train/metrics.py) |
| SR bank or height coordinates | [bank validation](../src/data/bank.py), [SR assembly](../src/build/sr.py) |
| Checkpoints or resume | [run lifecycle](../src/train/run/), [training state](../src/train/state.py) |
| Generation, tiling, or SR | [public API](../src/api.py), [src/predict/](../src/predict/) |
| Web behavior | [backend app](../backend/src/app.py), [frontend](../frontend/src/) |
| Metrics or synthetic data | [src/evaluate/](../src/evaluate/), [simul/](../simul/) |

## How the pieces fit

- Training: `run_train_1st.py` / `run_train_2nd.py` → `src/train/run/` → `src/build/`
  assembles data, models, and the trainer.
- `Trainer.step` owns optimizer and EMA updates. `Trainer.settings` holds runtime and
  loss settings built once in `src/build/trainer.py`; read them there instead of copying.
- `Trainer.conditions` (`ConditionSampler`) samples real batches, anchors, replay,
  dropout, and SR coarse inputs. Its replay state is saved through `train/state.py`.
- `train/loss/denoiser.py` owns the generator objective; `train/loss/gan.py` owns the
  plane, connectivity, and seam critics and their R1/R2 penalties.
- Generation: [TileSampler](../src/predict/tiling/sampler.py) owns state, noise, diffusion,
  fusion, and output. LR and SR tiling only provide their own conditions. Memory
  estimates in `src/predict/memory.py` and `src/predict/sr/memory.py` run before sampling.

## LR anchor losses

All three use the selected transition's x0 prediction at every t, follow the anchor
ramp, and are LR-only.

- **Continuity** (`loss.anchor_continuity_weight`, default `0.1`): next to each measured
  anchor, the phase-pair statistics up to `loss.connectivity.max_slice_gap` must match
  the measured 2D transition statistics. Only TV excess above real crop-to-crop
  variation is penalized.
- **Neighborhood** (`loss.anchor_neighborhood_weight`, default `0.1`,
  `loss.anchor_neighborhood_radius` = `2`): adjacent-plane statistics near an anchor
  must match those far from it in the same volume.
- **Seam critic** (`loss.anchor_seam_weight`, default `0`, off; `loss.anchor_seam_rows`
  = `4`): a 2D critic treats generated bands far from anchors as real and bands next to
  anchors as fake. Its "real" data is generated, not measured.

## Contracts to preserve

- Volumes use `(z, y, x)`. Crop and height coordinates are source pixels; LR/HR grids
  are voxels.
- `config/` holds presets, `src/config/` validates them, `src/build/` builds objects.
  Relative YAML paths resolve from the project root.
- LR inference reads the run's `train.yaml` and `config/gen.yaml`; SR reads settings
  inside its export; resume reads checkpoint settings.
- `src/train/state.py` owns training checkpoint formats and the resume config check;
  `src/train/run/resume.py` loads and relocates checkpoints, and `src/train/run/source.py`
  checks the frozen LR source. A relocated resume saves the moved paths in `train.yaml`.
- Keep measured 2D observations and generated replay references apart in losses,
  metadata, and claims.
- Folders: simulated data in `data/generated/<name>/`, inspection results in `checks/`,
  trained runs only in `run/`.

## Synthetic data

`python simul/run.py --config <file>` writes 3D TIFFs, per-axis slice PNGs, and
`simulation.yaml` (targets and measured fractions). The output folder must be empty.

- Default `kind: particles` ([geometry.py](../simul/src/geometry.py)): elongated phase 2
  and spherical phase 1 particles with periodic y/x.
- `kind: anode` ([anode.py](../simul/src/anode.py), [config_anode.yaml](../simul/config_anode.yaml)):
  tilted graphite flakes (1) and silicon spheres (2) in pores (0). A `*_vf` list sets
  fractions at evenly spaced heights, enforced per slice. Order bias alone cannot
  concentrate silicon once graphite fills the pores.

Train on the anode set with [config/train/anode.yaml](../config/train/anode.yaml) (LR) and
[config/train/anode_sr.yaml](../config/train/anode_sr.yaml) (SR): the stock presets with this
data and height conditioning, which SR must share with its LR source.

```bash
python simul/run.py --config simul/config_anode.yaml
python scripts/01_check_dataset.py --config config/train/anode.yaml --no-view
python run_train_1st.py --config config/train/anode.yaml --run-dir "run/anode-lr"
python scripts/02_check_generated.py --weight "run/anode-lr" --no-view
python run_train_2nd.py --config config/train/anode_sr.yaml --base-weights "run/anode-lr" --run-dir "run/anode-sr"
python scripts/06_check_hr.py --weight "run/anode-sr" --no-view
```

## Checks

```bash
python -m pip install pytest ruff httpx
python -m pytest -q
python -m ruff check .
```

Frontend, from `frontend/`: `npm ci`, `npm test`, `npm run build`. CUDA and MPS tests
skip without that hardware, so CPU passes do not cover AMP. For algorithm changes, also
compare samples at fixed seeds.

[Back to README](../README.md)
