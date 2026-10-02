# Development

## Where to make a change

| Change | Start here |
| --- | --- |
| Input images or resolutions | [config/data/](../config/data/), [RealDataset](../src/data/dataset.py) |
| Network architecture | [src/model/](../src/model/), [model assembly](../src/build/model.py) |
| Training step or objective | [Trainer](../src/train/trainer.py), [losses](../src/train/loss/) |
| SR bank fractions or height coordinates | [bank validation](../src/data/bank.py), [SR assembly](../src/build/sr.py) |
| Logging, checkpoints, or resume | [run lifecycle](../src/train/run/), [training state](../src/train/state.py) |
| Generation, tiling, or SR | [public API](../src/api.py), [src/predict/](../src/predict/) |
| Existing-volume conditioning | [VolumeCondition](../src/predict/volume_condition.py); [extend_hr](../src/predict/sr/extension.py) coordinates LR/SR extension |
| HTTP or web behavior | [backend app](../backend/src/app.py), [frontend](../frontend/src/) |
| Metrics or synthetic data | [src/evaluate/](../src/evaluate/), [simul/](../simul/) |

Training runs from `run_train_1st.py` / `run_train_2nd.py` through `src/train/run/`,
then `src/build/` assembles data, models, and the trainer. `Trainer.step` owns updates;
`train/loss/denoiser.py` owns the generator objective, and `train/loss/gan.py` owns
the shared plane/connectivity critic objective and R1/R2 input penalties.
The web backend calls `LowResolutionAPI`.

`Trainer.settings` holds runtime settings; its `loss` field holds the generator
objective settings assembled once in `src/build/trainer.py`. Read settings from
these objects instead of copying them onto the trainer or rebuilding them per step.
`Trainer.cfg` keeps the resolved configuration used for logging and checkpoints.

## Contracts to preserve

- Volumes use `(z, y, x)` axes. Crop/height coordinates use source pixels;
  LR/HR grids use voxels. `DataResolution` names these units explicitly.
- `config/` holds user presets, `src/config/` resolves and validates them,
  and `src/build/` constructs runtime objects. Backend config contains HTTP limits.
- LR inference reads the run's `train.yaml` and `config/gen.yaml` generation defaults.
  SR reads settings embedded in its weight export; resume reads checkpoint settings.
- `src/train/state.py` owns checkpoint formats. Preserve frozen-LR source checks
  during creation, refresh, and resume, and saved-bank integrity checks on resume.
- SR bank height coordinates use source pixels. Bank validation resolves shared
  extents to one value per sample and keeps existing per-sample tensors connected
  to bank refresh updates.
- Measured 2D observations and generated replay references have distinct roles.
  Preserve that distinction in losses, metadata, and evaluation claims.

Tiled fusion still uses buffers spanning the output's H/W plane; memory estimates
include them. A disconnected web client does not cancel ongoing model computation.

## Checks

Run in the project environment, from the repository root:

```bash
python -m pip install pytest ruff httpx
python -m pytest -q
python -m ruff check .
```

Frontend checks, from `frontend/`: `npm ci`, `npm test`, and `npm run build`.
CUDA tests skip when a GPU is unavailable. Passing CPU tests does not establish
CUDA/AMP behavior or trained reconstruction quality. For algorithm changes, compare
training state and generated samples at fixed seeds as well as running regression tests.

[Back to README](../README.md)
