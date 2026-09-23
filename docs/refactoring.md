# Refactoring plan and evidence

This work preserves the public inference API, CLI, resolved training settings,
checkpoint formats, and source-pixel coordinate conventions. Algorithm changes
require a reproducible defect or a documented mathematical justification.

## Current boundaries

| Area | Current responsibility | Review focus |
| --- | --- | --- |
| `src/data`, `src/prepare`, `src/plane.py` | Source labels, crop metadata, phase fractions and coordinates | Keep labels distinct from probabilities and preserve z coordinates. |
| `src/config`, `src/build` | Validate settings and assemble models, data and trainers | Keep validation consistent between LR, SR and resumed runs. |
| `src/model` | Denoiser, critics and diffusion process | Check transition indexing, gradients and conditioning contracts. |
| `src/train/trainer.py` | Batch preparation, condition sampling, losses, optimizer updates and metrics | Separate responsibilities that can be tested independently; retain one visible step order. |
| `src/train/run`, `src/train/state.py` | Execution, LR bank publication, checkpoint capture and resume | Preserve safe checkpoint boundaries and immutable published banks. |
| `src/predict` | Generation, tiled fusion, SR, extension and memory planning | Keep chunked conversion and CPU/GPU storage decisions consistent. |
| `backend/src`, `frontend/src` | HTTP limits, response lifetime, request state and display | Preserve raw response shape/length and stale-request protection. |

The existing top-level boundaries are useful. A new folder hierarchy is not an
end in itself. The original training coordinator had approximately 1,500 lines
and also implemented individual loss terms and condition construction. Extract
behavior only when its inputs, outputs and owner become clearer; avoid splitting
the class into mixins with implicit shared state.

## Work sequence

1. Record existing changes and run the existing Python and frontend checks.
2. Review training execution and algorithms with Chat MCP, then verify findings
   against callers and regression tests. Large review scopes must be split to
   fit the connector's context limits.
3. Refactor coherent training responsibilities and reproduce confirmed defects
   before fixing them. Keep LR/SR objective weights and gradient flow unchanged
   except for documented corrections.
4. Review inference, tiling, memory, backend and frontend contracts. Apply the
   same evidence requirement to changes in these areas.
5. Review the resulting changes with Chat MCP, run the relevant integration
   checks, and record remaining risks and checks that could not run.

## Baseline and constraints

- Work started with existing edits to `src/storage.py` and
  `tests/test_storage.py`; those edits are outside this refactor's ownership.
- The repository `.venv` is usable. CUDA is unavailable in this environment;
  CPU tests cannot establish CUDA/AMP behavior.
- Baseline: 1,026 Python tests passed, 16 skipped, and 9 subtests passed. Ruff,
  all 29 frontend tests, and the frontend production build passed.
- Long training runs, experimental preset changes and checkpoint migrations
  are not part of routine structural validation.

## Implemented training boundaries

- `src/train/loss/denoiser.py` owns the differentiable generator objective:
  grouped adversarial losses, anchor/VF losses, replay and measured-transition
  weights, height profiles and SR coarse consistency. It takes explicit batches,
  models and immutable loss settings, and returns losses and diagnostics.
- `Trainer.update_denoiser` owns freezing critics, autocast, backward, optimizer
  stepping and update counters. EMA and checkpoint ownership remain unchanged.
  It detaches the result only after backward. `src/train/step.py` already owns
  finite-loss/gradient checks, so no parallel optimization framework was added.
- `Trainer.prepare_step` selects domains and acquires real batches once, then
  dispatches to `prepare_lr_step` or `prepare_sr_step`. Stage-specific condition
  assembly stays with its existing state owner; no mixins or shared context
  object were introduced. The original-trainer comparison still passes.
- `src/train/loss/gan.py` shares critic input dispatch and active group selection
  between discriminator and generator objectives, preserving their weighting.
- `src/train/run/source.py` owns frozen LR source hashes and source configuration
  loading. Bank generation and refresh use the same source validation policy.
  Configuration parsing stays in `src/config`; relocation stays in training.

## Confirmed defects

1. Anchor-focused slice sampling stopped at the first randomly selected inactive
   batch or non-intersecting anchor region. Later valid candidates could be
   ignored. Sampling now chooses from eligible batch/plane/region intersections.
   Regression tests verify actual sampled values, partial batch activation,
   later valid regions, and zero gradients on excluded measured planes.
2. Initial SR bank creation did not recheck the frozen source hashes recorded
   during source preparation. It now rejects changed weights/configuration
   before loading the generator, as bank refresh already did.
3. Bank refresh passed an external data-YAML string to `PathRemapper.config`,
   which expects resolved data settings. Initial source preparation accepted
   this configuration. Shared parsing now expands external settings and applies
   relocation before split normalization. A relocated external YAML reference
   is mapped before opening it, with foreign Windows/POSIX roots handled before
   host-specific path resolution. The original source YAML is never rewritten.

The sampling and source failures were reproduced before their fixes. Initial
focused checks passed: 154 data/prepare/diffusion/trainer tests (4 skips), 83
source/configuration/integration tests, and 211 training/SR/stability tests
(3 CUDA skips). Three additional isolated objective tests verify grouped loss
weights, gradients with diagnostics enabled/disabled, and differentiable zero
loss for empty sampled axes.

Follow-up source checks passed all seven regression cases, including both
weights/config hashes, ordinary/source loader equivalence and cross-platform
external-YAML relocation. The broader source integration selection passed 86
tests before the final two cross-platform cases were added.

A local comparison against the original trainer ran three steps each for
height-conditioned LR, LR with spatial profiles, and SR. Metrics and captured
model/EMA/optimizer/scaler/counter state were exactly equal, with zero tolerance.
The comparison also passed with diagnostics enabled on every step and both
intermediate and final diffusion transitions.
This checks the loss extraction, not statistical equivalence of the intentionally
corrected slice sampler or GPU execution.

## Review decisions and remaining work

Chat MCP reviewed training execution and the proposed loss boundaries. Its
initial execution review exceeded the connector wait limit; the answer was
recovered during cancellation and also supplied by the user. Findings above
were checked locally before implementation.

The post-change loss review found no actionable defects in the supplied objective,
GAN helpers, batch result and objective tests. The connector's working-tree
review did not include ignored local before/after snapshots; local equivalence
checks provide the before/after evidence.

Integration check before the source-snapshot follow-up: `pytest -q -rs` passed 1,038
tests and 9 subtests, with 16 skips all explicitly attributed to unavailable
CUDA. Repository-wide Ruff and `git diff --check` passed. The five warnings are
the same dependency deprecations and KID feature-buffer notices seen at baseline.
Frontend source was not changed; its baseline 29 tests and production build
passed. No CUDA/AMP or long-run reconstruction-quality result is claimed.

The source/sampling review raised two further source-integrity concerns: an
external data YAML was not covered by `config_sha256`, and the generator and
bank conditions could read different configurations. Local tracing also showed
that `load_generator` reopened the original config after the relocation-aware
loader, undoing external-YAML relocation. Earlier isolated tests mocked that
loader, so they did not cover this failure. Follow-up work uses one resolved
config to build the generator and conditioning datasets, and adds real-builder
regressions plus an optional external-data digest without rewriting legacy
checkpoints. This follow-up is not covered by the 1,038-test result above.

Source-snapshot implementation and regression checks are now complete. Both
creation and refresh use `load_frozen_source` and pass its resolved config to
the real generator builder. New external-YAML sources record `data_sha256`;
legacy sources without that optional field remain readable. The combined
source/configuration/inference selection passed 413 tests, skipped 8, and failed
one stochastic measured-transition assertion; that test passed when rerun alone.
This is not reported as an entirely passing combined run. Ruff and diff checks
passed after the source changes.
The post-change source review (`refactor-source-final-1790154261468`, conversation
`2f5e03e0-5987-4308-a337-3d5fb903a283`) hit `PAGE_HIDDEN`. One recovery attempt
failed; cancellation was confirmed and returned only an incomplete fragment.
This external review is therefore incomplete, not a clean review result.

The stochastic assertion was subsequently reproduced with isolated seed 1:
the replay z interval was [0, 8), while new measured crops started at z=9 or
later. No rows matched, so zero measured-transition loss is correct. The test
now explicitly exercises positive matches (seed 0) and no matches (seed 1),
restores RNG state, and retains the prohibition on scoring replay volumes as
measured references. All 17 targeted transition/conditioning cases passed.
No training algorithm change was needed for this failure.

`InferenceAPI` also now passes its single resolved config to `build_generator`,
so crop metadata and model construction use the same settings. Its targeted
inference/backend checks passed 95 tests with one CUDA skip. The separated
LR/SR preparation paths passed 104 training/SR/objective tests with one CUDA skip.

The review's proposed masking of gradients at orthogonal anchor intersections
was not adopted: the existing sampling contract excludes parallel measured
planes, while orthogonal anchor-focused crops intentionally include the
intersection. Expanding exclusion to individual voxels would change the
adversarial training objective and needs a separate algorithmic justification.

The suggested diagnostic `fake_curr.detach()` defect was rejected after tracing
`generate_pair`: the preceding diffusion chain already runs under `no_grad` and
explicitly detaches `current`. Diagnostics intentionally use a separate leaf;
changing that graph would change the training algorithm. Penalizing only the
previous input in R1/R2 was not changed without an objective-level justification.

Additional empty-dataset checks in bank sampling were not added: source height
validation, image collection and dataset construction already reject missing
side images and empty image groups on the supported path.

The tiling review was supplied by the user, then recovered from the original
request with confirmed cancelled status. Local checks distinguish three points:

- Fusion remains a tile-depth circular slab across the entire H/W plane. Its
  storage is `4 * (C + 1) * min(tile_size, D) * H * W` bytes. This is a real
  scalability limit, but it is already included in `estimate_memory` and both
  RAM/CUDA budget checks. Two full FP16 diffusion states also remain necessary
  in the current implementation. For C=3, shape=2048 cubed and tile_size=224,
  these states require 96 GiB and fusion requires 14 GiB before other buffers.
  Flushing y/x immediately would discard contributions from later z layers.
  A smaller accumulator needs an explicit traversal/storage/recomputation
  tradeoff and numerical equivalence evidence; it is not a local bug fix.
- With tile_size=8, overlap=2 and axis length=13, starts are (0, 4, 5): the last
  overlap spans 7 voxels. Fixed boundary tapering leaves a wider equal-weight
  plateau, but normalized fusion still covers voxels correctly. No output
  defect was established. Changing ramp widths is a blending-policy change,
  so the existing weights are preserved.
- `TilePlan` now retains generation-space starts used to derive its grid,
  seams and tiles. Direct construction remains supported. Output-space stats
  retain the padded grid for reporting and are explicitly rejected as input
  to tile generation when their shape differs from the generation shape.
  Base preservation order showed no confirmed defect.

After the geometry cleanup, inference/tiling, SR and scale-script checks passed
268 tests with 9 CUDA skips. Repository-wide Ruff passed. These checks include
circular-slab versus full weighted fusion, bounded SR fusion, base preservation,
snapped geometry and output-statistics misuse.

The source-change Chat MCP verdict was recovered on the same request ID in a
later turn. Both follow-up corrections are implemented and verified below.
Fusion's H/W scaling remains a documented limitation.

Chat MCP completed the HTTP/UI review (`refactor-http-ui-1790154661035`). Its
two observations were checked against the actual application wiring:

- Synchronous generation does continue after client disconnect until inference
  returns. This consumes the existing generation/download slots until cleanup;
  it does not bypass their bounds. Cancellation of computation is not currently
  part of the API. Cooperative cancellation would need checks through both
  ordinary and tiled diffusion plus safe RNG/lock cleanup. This is a remaining
  service capability, not a reason to release locks while generation runs.
- Input invalidation gates stale results but does not abort an individual
  request. `Sidebar.vue` disables all inputs during generation and `App.vue`
  disables cropping. The suggested overlapping user-request path is therefore
  unavailable through the current UI. Unmount already aborts network requests,
  and tests cover stale-result rejection and clearing busy state. The current
  component/request boundary is retained; no cancellation UI was added.

The reviewed boundaries remain HTTP validation/preflight in `backend/src/app.py`
and `schema.py`, transfer ownership in `response.py`, raw protocol parsing in
`frontend/src/api.js`, and request/revision state in `use-generation.js` and
`input-state.js`. Existing response tests exercise completion, disconnect,
cancellation and send errors for raw/TIFF transfers, including slot cleanup.

A source-review retry could not reuse the cancelled conversation (`INVALID_HANDLE`).
A fresh attempt (`refactor-source-final2-1790154981371`) returned
`BRIDGE_UNAVAILABLE`; lookup returned `NOT_FOUND`, so there is no confirmed live
review to wait on at that point. The subsequent recovery is recorded below.

Recovery update: retrying the original missing request with identical inputs
completed successfully (`refactor-source-final2-1790154981371`, conversation
`f7293668-19ea-460d-a553-0dd6c630530d`). Its three findings were checked locally:

- Resume checked the saved bank digest but did not recheck frozen LR source
  weights/config/external-data hashes before training. This conflicts with
  the documented frozen-source resume contract. Resume now checks the bank
  digest, validates the source using accumulated path mappings, and only then
  loads the bank. Validation precedes creation of the new run directory.
- Publishing a refreshed bank copied the previous bank path/digest into its
  internal source provenance. A regression reproduced the stale fields;
  publication now excludes `bank` and `bank_sha256` from that metadata while
  preserving the newly published bank reference in the run configuration.
- The proposed same-run step collision is not a supported resume path:
  `make_run_dir` requires a new directory (`exist_ok=False`) on every resume,
  as documented in README. Immutable step publication remains unchanged.

The metadata regression failed before its fix. Source regressions cover changed
weights, configuration and external YAML, plus accumulated relocation maps.
An integration selection passed 420 tests with 8 skips but exposed one existing
error-priority assertion: when both source and bank changed, the bank error must
remain first. Validation was reordered to preserve that priority. The final
affected selection (`test_bank_source.py`, `test_sr_pipeline.py`,
`test_relocate.py`, and repeated height-conditioned SR relocation) passed all
33 tests. Ruff and diff checks passed. No unresolved test failure remains.

The critic-update boundary assessment retains the existing update methods:
they own real/fake pair preparation, regularization cadence, weighted backward
and optimizer counters. Pure score dispatch/group selection already moved to
`loss/gan.py`, and finite-gradient/optimizer behavior remains in `train/step.py`.
Extracting the remaining loop would require passing trainer-owned augmentation,
diffusion, profile sampling and optimizer state without a demonstrated benefit.
Existing tests cover grouped weights, owned/shared domains, real batch reuse,
single-axis updates and finite-gradient handling; the original/current trainer
comparison also covers critic parameters and optimizer state.

The preparation assessment found no additional confirmed defect. The adopted
boundary is common domain/critic selection and one-time real batch acquisition,
followed by separate LR and SR `StepPreparation` assembly. Keep anchor-bank
alternation with the anchor lifecycle, preserve measured-anchor VF precedence,
and use the augmented clean LR bank sample as the SR consistency target while
feeding its corrupted counterpart to the model. Existing tests protect real
batch reuse, VF/profile precedence, source-height metadata and bank immutability.

## Current completion audit

| Requirement | Evidence and status |
| --- | --- |
| Map LR/SR, data, model, inference and HTTP/UI responsibilities | Boundary map above; training/design, source, tiling and HTTP/UI Chat MCP reviews checked against local callers. Complete. |
| Refactor for explicit ownership and remove relevant duplication | Pure denoiser objective, separate LR/SR preparation, shared source loader/model builder, shared tile starts. Complete. |
| Correct demonstrated defects without changing unrelated algorithms | Anchor sampling and frozen-source regressions; measured-transition zero-match case reproduced and test corrected. Complete. |
| Preserve interfaces, coordinates, state and memory contracts | Entire Python suite, tiny real LR bank generation/refresh, exact original/current trainer comparison, tiling full-fusion equivalence, existing backend protocol and cleanup tests. CPU evidence complete. |
| Verify the final working tree | Broad integration before the last source corrections: 1,044 passed, 16 CUDA skips, 9 subtests passed. After those corrections, all 33 affected tests passed. Repository-wide Ruff plus final changed-file Ruff and diff checks passed. Frontend unchanged; baseline 29 tests and production build passed. Complete. |
| Final external review of source follow-up | Recovered successfully; two findings fixed and verified, one rejected against the new-run resume contract. Complete. |
| Document decisions and remaining risk | This document records rejected findings, CPU-only coverage, fusion scaling, disconnect behavior and the recovered external review. Complete. |

The final integration run took 203.50 seconds. All 16 skips explicitly require
CUDA; the five warnings are existing dependency deprecations/KID buffer warnings.
No trained reconstruction-quality, long-run convergence or CUDA/AMP result is
claimed. The user's pre-existing storage changes were outside edit ownership;
no experiment configuration, weights or run outputs were replaced.

The refactoring goal is complete. Remaining items are documented limits
and future capability/quality work, not unimplemented fixes from this review.
