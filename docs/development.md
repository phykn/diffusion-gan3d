# Development

## Where to make a change

| Change | Start here |
| --- | --- |
| Input images or resolutions | [config/data/](../config/data/), [RealDataset](../src/data/dataset.py) |
| Network architecture | [src/model/](../src/model/), [model assembly](../src/build/model.py) |
| Training step or objective | [Trainer](../src/train/trainer.py), [losses](../src/train/loss/) |
| Training data, anchors, or condition sampling | [ConditionSampler](../src/train/conditioning.py), [step inputs](../src/train/batch.py) |
| Plane pairs, connectivity triplets, or seam bands | [section sampling](../src/data/slice.py) |
| Step metrics and TensorBoard tags | [metrics](../src/train/metrics.py) |
| SR bank fractions or height coordinates | [bank validation](../src/data/bank.py), [SR assembly](../src/build/sr.py) |
| Logging, checkpoints, or resume | [run lifecycle](../src/train/run/), [training state](../src/train/state.py) |
| Generation, tiling, or SR | [public API](../src/api.py), [src/predict/](../src/predict/) |
| Scoped inference or evaluation seeds | [seeded_rng](../src/random.py) |
| Existing-volume conditioning | [VolumeCondition](../src/predict/volume_condition.py); [extend_hr](../src/predict/sr/extension.py) coordinates LR/SR extension |
| HTTP or web behavior | [backend app](../backend/src/app.py), [frontend](../frontend/src/) |
| Metrics or synthetic data | [src/evaluate/](../src/evaluate/), [simul/](../simul/) |

Training runs from `run_train_1st.py` / `run_train_2nd.py` through `src/train/run/`,
then `src/build/` assembles data, models, and the trainer.
`train/loss/denoiser.py` owns the generator objective, and `train/loss/gan.py` owns
the shared plane, connectivity, and seam critic objective and R1/R2 input penalties.
The web backend calls `LowResolutionAPI`.

`Trainer.step` owns optimizer and EMA updates. `collect_metrics` in
`src/train/metrics.py` assembles the step result, converts tensor diagnostics to
plain values, and marks unavailable anchor-boundary measurements. It does not
update training state. The run loop writes every metric and diagnostic to
TensorBoard, including zero-valued inactive metrics, except where a section below
says an objective records tags only while enabled. Unavailable measurements
are recorded as NaN gaps. Experiment scripts use `scripts/common/metrics.py` to
read the full scalar history by step, converting those gaps back to `None`.

`Trainer.settings` holds runtime settings; its `loss` field holds the generator
objective settings assembled once in `src/build/trainer.py`. Read settings from
these objects instead of copying them onto the trainer or rebuilding them per step.
`Trainer.cfg` keeps the resolved configuration used for logging and checkpoints.

`Trainer.conditions` is a `ConditionSampler` sharing those settings and data
streams. It samples real batches, LR anchors and replay, condition dropout, and
SR coarse inputs. `Trainer.prepare_step` selects the active critics and asks the
sampler for a `StepPreparation`, including sampling diagnostics. Checkpoint state
includes the sampler's replay bank and next-anchor choice through `train/state.py`.

[TiledGenerator](../src/predict/tiling/generator.py) prepares LR geometry,
anchors, height, and phase profiles. [SR tiling](../src/predict/sr/tiled.py)
prepares coarse inputs and height. Both call
[TileSampler](../src/predict/tiling/sampler.py), which owns state allocation,
noise initialization, diffusion, fusion, and the final CPU output. Changes to
that lifecycle belong in the sampler; stage-specific conditions stay with their
provider. Existing-volume conditioning belongs to `VolumeCondition`.

[LR memory estimates](../src/predict/memory.py) and
[SR memory estimates](../src/predict/sr/memory.py) are checked before sampling.
They share tile-buffer and model-workspace calculations, using the actual tile
shape; SR additionally budgets coarse interpolation and label conversion.
`src/random.py` scopes seeded inference and KID evaluation to the CPU and
selected CUDA or MPS device, restoring the caller's random state on exit;
`preserved_rng` replays the same noise for same-noise baselines.

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

## LR anchor neighborhood transitions

`loss.anchor_neighborhood_weight` controls the same-volume neighborhood objective
(default `0.1`). `loss.anchor_neighborhood_radius` is the number of adjacent pair
layers on each normal side (default `2`). Distance `0` includes the anchor plane;
distance `1` compares its immediate neighbor with the next plane. Distances use
the closest endpoint to a visible anchor, in LR voxels. Both settings are LR-only.

The objective runs on the selected transition's x0 prediction at every t and uses
the anchor ramp. Measured and replay anchors both define neighborhoods. Fixed
endpoints use detached condition fractions; only voxel pairs with both endpoints
fixed are excluded. Overlapping regions count each voxel pair once per axis.
Partial anchors restrict statistics to their projected coverage. Reference pairs
are outside all same-axis neighborhoods within that coverage, separately for each
volume and axis; an axis without reference pairs contributes nothing.

Each adjacent plane pair supplies an oriented joint phase matrix. The detached
reference is the voxel-count-weighted mean of far-pair matrices. The allowance is
their weighted mean TV against leave-one-out reference means; a single reference
pair has zero allowance. The loss is positive TV excess above that allowance,
with a float32 epsilon at the boundary. Pair losses are weighted by valid voxel
counts within each distance; available distances are averaged equally.

TensorBoard records `loss/anchor_neighborhood`,
`anchor_neighborhood/contribution`, and
`anchor_neighborhood/distanceN/{loss,tv,tolerance,pairs}` even when the weight is
zero. `far_pairs` counts reference voxel pairs and `reference_axes` counts supported
volume-axis combinations. Zero pair counts mark unavailable distances. A zero
weight avoids constructing an additional gradient graph for these diagnostics.

## LR anchor seam critic

`loss.anchor_seam_weight` enables an adversarial seam objective (default `0`, off);
`loss.anchor_seam_rows` is the band height in LR voxels (default `4`). Both are LR-only.
`sample_seam_bands` in `src/data/slice.py` cuts bands of that many rows from the
selected transition's x0 prediction, starting next to each visible anchor on both
normal sides and running away from it. Bands are `patch_size // 2` voxels wide and
must avoid fixed voxels. Far bands come from the same prediction, more than `rows`
voxels from every same-axis anchor.

`SeamCritic2D` scores a band from its section plane, diffusion time, and domain. It
treats far bands as real and near bands as fake, so it compares generated content
next to an anchor with generated content elsewhere, not with measured 3D data. The
generator receives its adversarial loss on near bands, scaled by the anchor ramp.
Near and far bands are drawn with equal counts for each anchor axis and section
plane, at most `slice_pairs_per_plane` per plane, so neither the plane label nor the
row orientation can tell them apart. With height conditioning, bands whose rows run
along z are read upward and the width direction is never mirrored across z.

The LR seam critic is always built, so `critic_s.pt` is saved and LR checkpoints hold
`seam` and `seam_optim` even at weight `0`. Unlike the neighborhood diagnostics, seam
tags are recorded only while the weight is positive: `loss/anchor_seam`,
`loss/critic_anchor_seam`,
`anchor_seam/{contribution,near_bands,far_bands,near_score,far_score,r1}`, and
`r2/anchor_seam` when R2 applies. Regression tests are in
`tests/train/test_anchor_seam.py`.

## Contracts to preserve

- Volumes use `(z, y, x)` axes. Crop/height coordinates use source pixels;
  LR/HR grids use voxels. `DataResolution` names these units explicitly.
- `config/` holds user presets, `src/config/` resolves and validates them,
  and `src/build/` constructs runtime objects. Backend config contains HTTP limits.
  Both training stages validate shared numeric settings in `src/config/values.py`.
  `validate_training_settings` checks settings without defaults and batch-size
  constraints before data or models are built; `validate_sr_config` also resolves SR
  geometry and validates SR source/bank settings. Relative paths in YAML resolve
  against the project root, including `train.initial_weights`.
- LR inference reads the run's `train.yaml` and `config/gen.yaml` generation defaults.
  SR reads settings embedded in its weight export; resume reads checkpoint settings.
- `src/train/state.py` owns checkpoint formats. Preserve frozen-LR source checks
  during creation, refresh, and resume, and saved-bank integrity checks on resume.
- Plane critics use only a global head in LR and SR. Connectivity keeps both
  heads; `loss.connectivity_local_weight` scales its local adversarial and R1/R2
  terms. TensorBoard records the unweighted heads separately as
  `loss/{generator,critic}_connectivity_{global,local}`; the existing combined
  connectivity tags retain the weighted sum, before generator ramp/outer weighting.
- Connectivity triplets are sampled when `loss.connectivity.adversarial_weight`
  is positive and an eligible multi-anchor replay is available at the final
  diffusion step. The real transition objective uses measured 2D statistics
  independently of these triplets.
- SR bank height coordinates use source pixels. Bank validation resolves shared
  extents to one value per sample and keeps existing per-sample tensors connected
  to bank refresh updates.
- Measured 2D observations and generated replay references have distinct roles.
  Preserve that distinction in losses, metadata, and evaluation claims. The seam
  critic's "real" bands are generated far bands, not measured data.
- `simul/` packs particles with periodic y/x boundaries (and z when the radius has no
  height gradient), so the exported volumes have no boundary density profile. Random
  packing can jam below the configured fractions; `simulation.yaml` records the
  measured fractions and the optional `output.seed`.

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
CUDA tests skip without a CUDA GPU and Apple GPU tests skip without MPS. Mixed
precision runs only on CUDA. Passing CPU tests does not establish CUDA/AMP behavior
or trained reconstruction quality. For algorithm changes, compare
training state and generated samples at fixed seeds as well as running regression tests.

[Back to README](../README.md)
