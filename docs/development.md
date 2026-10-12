# Development

## Where to make a change

| Change | Start here |
| --- | --- |
| Input images or resolutions | [config/data/](../config/data/), [RealDataset](../src/data/dataset.py) |
| Network architecture | [src/model/](../src/model/), [model assembly](../src/build/model.py) |
| Training step or objective | [Trainer](../src/train/trainer.py), [losses](../src/train/loss/) |
| Step metrics and TensorBoard tags | [metrics](../src/train/metrics.py) |
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

`Trainer.step` owns optimizer and EMA updates. `collect_metrics` in
`src/train/metrics.py` assembles the step result, converts tensor diagnostics to
plain values, and marks unavailable anchor-boundary measurements. It does not
update training state. The run loop writes every metric and diagnostic to
TensorBoard, including zero-valued inactive metrics. Unavailable measurements
are recorded as NaN gaps. Experiment scripts use `scripts/common/metrics.py` to
read the full scalar history by step, converting those gaps back to `None`.

`Trainer.settings` holds runtime settings; its `loss` field holds the generator
objective settings assembled once in `src/build/trainer.py`. Read settings from
these objects instead of copying them onto the trainer or rebuilding them per step.
`Trainer.cfg` keeps the resolved configuration used for logging and checkpoints.

## LR anchor continuity

`loss.anchor_continuity_weight` enables an additional LR generator loss (default
`0.1`; `0` disables it). Its contribution uses the existing anchor ramp and applies
to the selected transition's **x0 prediction at every t**. The preceding sampled
chain stays detached. Model architecture, inference, and the existing objectives
are unchanged; SR does not accept this setting.

For each visible measured anchor, the loss compares neighbors on both normal
sides at every LR-grid distance through `loss.connectivity.max_slice_gap`.
Replay supplies only its measured root, never generated planes as observations.
Destinations covered by any active anchor are excluded voxel by voxel. Partial
anchors and out-of-volume destinations are handled by the same valid-pair mask.

The observed row-wise phase-pair statistics in `loss/transition.py` are computed
once per step and shared with the existing real transition objective. Only
observations from the target domain containing the physical normal direction
supply targets; absent directions or source phases are skipped. No isotropy or
3D reference is invented. Without height conditioning, each real crop's joint
matrix is averaged with its transpose. With height conditioning, positive and
negative directions remain separate and use the same overlapping pair-midpoint
height intervals as the real transition loss. Negative directions transpose the
joint matrix **before** computing conditionals.

For valid pairs, `J[a,b] = mean(anchor[a] * prediction[b])` and
`pi[a] = mean(anchor[a])`. The target is `pi[a] * T[b|a]`, with `T` obtained by
row-normalizing the pooled real joint statistics. Thus the target retains the
anchor's own composition. All operations use phase fractions, without argmax.
The error is `TV = 0.5 * sum(abs(J - pi*T))`. Its allowance is the anchor-weighted
mean conditional TV deviation of individual real crops from the pooled target;
only excess above this allowance is penalized. Height-matched crops are weighted
by their overlapping row coverage; missing phase rows do not contribute to the
allowance. Valid pairs are averaged within each gap, then available gaps are
averaged equally. No additional reference generation or discriminator is needed.

TensorBoard records `loss/anchor_continuity`, its weighted `contribution`, and
`anchor_continuity/gapN/{loss,tv,tolerance,pairs,sharpness}`. Sharpness is
`sum(probability**2)` across phase channels: `1/C` for a uniform distribution and
`1` for a pure phase. `sharpness_near`, `sharpness_far`, and their signed
`sharpness_difference` compare the union of eligible neighbors with remaining
unconstrained voxels in the same visible batch items. These are diagnostics,
not extra losses. Missing near/far populations appear as NaN gaps with voxel
counts of zero. Pair counts can exceed unique voxel counts when anchors overlap.

CPU regressions cover detached, extruded, and transition-matched neighbors,
all transition times, replay/dropout masks, fractional phases, crop variation,
directional height matching, and TensorBoard output. They establish the objective
and gradient behavior, not trained 3D reconstruction quality. Existing weights
need further training for this loss to affect generation.

## Contracts to preserve

- Volumes use `(z, y, x)` axes. Crop/height coordinates use source pixels;
  LR/HR grids use voxels. `DataResolution` names these units explicitly.
- `config/` holds user presets, `src/config/` resolves and validates them,
  and `src/build/` constructs runtime objects. Backend config contains HTTP limits.
  Both training stages validate shared numeric settings in `src/config/values.py`;
  `validate_sr_config` also resolves SR geometry, requires training fields, and
  validates SR source/bank settings.
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
