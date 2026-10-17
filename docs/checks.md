# Run and inspect

Run from the project root with the environment activated. `--weight` accepts a run
directory, its `generator.pt`, or a step-specific EMA export under `checkpoints/`.
Each inspection saves results in `run/checks/` and opens a viewer; add `--no-view`
for files only.

| Script | Purpose | Weights |
| --- | --- | --- |
| [01_check_dataset.py](../scripts/01_check_dataset.py) | Preview training crops | Optional; uses the LR preset when omitted |
| [02_check_generated.py](../scripts/02_check_generated.py) | Generate an LR volume | LR |
| [03_check_anchor.py](../scripts/03_check_anchor.py) | Compare multiple anchor conditions | LR |
| [04_check_scale_up.py](../scripts/04_check_scale_up.py) | Inspect tiled generation and seams | LR |
| [05_check_continuation.py](../scripts/05_check_continuation.py) | Grow a volume from a boundary section | LR |
| [06_check_hr.py](../scripts/06_check_hr.py) | Compare LR, upsampling, and SR | SR; reads the saved LR source |

```bash
python scripts/04_check_scale_up.py --weight "run/my-lr-run" --blocks 2 2 2 --no-view
python scripts/05_check_continuation.py --weight "run/my-lr-run" --anchor image.png --axis 0
```

`--blocks D H W` sets the output size along `(z, y, x)` to
`patch + (n - 1) * (patch - 2 * overlap)` voxels. The tile grid also covers the context
margin (the denoiser's downsampling factor, 8 voxels for the preset channels), so it can
hold more tiles than the block count: a 64³ model with overlap 8 turns `--blocks 2 2 2`
into a 112³ output from a 3 × 3 × 3 grid. The script prints the actual grid.
`--axis 0/1/2` selects `xy/xz/yz`.
Anchor demos use generated references by default; supply `--anchor` to script 05
for a measured boundary image. These comparisons do not establish measured 3D accuracy.

## Options and outputs

- `--device cpu|cuda|mps`, `--seed`, `--domain`: device (default: CUDA, then MPS, then CPU),
  repeatable sample, and dataset ID (`02–06`).
- `--napari`: open a 3D viewer (`02–06`). `--help` lists each script's full options.
- `--height-origin`, `--height-extent`: Z origin and full source height in **source pixels**.
  Height is the row direction in xz/yz images. Script 05 infers it from its input crop;
  set the extent explicitly when source heights differ.
- `01` saves a crop PNG; `02–05` save `volume.tiff`, `volume.png`, and `volume.json`;
  `05` also saves the slice strip `volume_continuation.png` (`--figure` moves it).
  `06` saves LR/HR volumes, `comparison.png`, `report.json`, and optional LR fractions.

LR guidance, anchor strength, and overlap default to [config/gen.yaml](../config/gen.yaml).
Anchor strength is a 0–1 logit interpolation, not a pixel-match percentage.
Script 06 uses SR guidance `1.0` unless overridden. Its saved LR source is hash-checked;
`--lr-weight` selects an alternative and records it as unverified in `report.json`.

## Save and resume

The run's `generator.pt` is for inference; root generator and critic exports are
overwritten at `weights_every_steps`. At `archive_every_steps`, `checkpoints/`
receives `step_00000100.pt` (training state, including optimizers) and
`generator_00000100.pt` (EMA inference weights) for step 100. Both files are closed
before `step_00000100.complete` is written; keep this completion marker for resume.
Existing files for the same step are never overwritten. Older checkpoint names
with numeric suffixes can still be loaded.

Completion and Ctrl+C at a completed step also save progress. `structure_every_steps`
controls diagnostics. Load exports after writing has finished. All training metrics
and diagnostics are stored in `tensorboard/`; view them with
`tensorboard --logdir run/my-lr-run/tensorboard`. No data manifest or JSONL training
log is written. Data fingerprints needed for resume remain in training checkpoints.

Resume requires the current checkpoint schema, including `path_maps` (an empty
list when no paths have moved). LR checkpoints also require the connectivity model,
optimizer, and update counter even when replay losses are disabled. Checkpoints
missing these fields are rejected; they are not upgraded during loading. The one
exception is the LR seam critic: a checkpoint saved before it existed resumes with a
fresh seam critic only while `loss.anchor_seam_weight` is `0`. A checkpoint written
without CUDA mixed precision (CPU or MPS) resumes on CUDA with a fresh AMP loss scale.
The first Ctrl+C stops after the current step; a second one stops at once without
saving that partial step.

```bash
python run_train_1st.py --resume "run/my-lr-run" --steps 20000
```

`--steps` is the total target, including completed steps. Resume uses saved settings
and creates a new run directory. Both training commands accept `--path-map OLD NEW`
when files move; repeat it for separate data/run roots. Copy the original data and
completed artifacts. SR also needs its unchanged frozen LR weights/config and saved bank.

## Anchor-neighborhood loss benchmark

The paired benchmark checks loss, gradients and diagnostics against the loop
reference from commit `0dce898`, then times forward and forward-plus-backward
separately. Inputs and anchors are created before timing; implementation order
alternates, and CUDA timings synchronize the device. Use a fresh output path.

```powershell
& .\.venv\Scripts\python.exe scripts/experiments/benchmark_anchor_neighborhood.py --device cpu --threads 1 --warmup 3 --repeats 21 --output run/checks/anchor-neighborhood/benchmark.json
```

On 2026-10-13, Windows / PyTorch 2.14.0 CPU, one thread, batch size 1 and three
phases gave these **forward-plus-backward medians** over 21 paired repeats:

| Volume | Anchor axes | Radius | Loop | Batched | Speedup |
| --- | --- | --- | --- | --- | --- |
| 32³ | xy | 2 | 45.48 ms | 6.03 ms | 7.54× |
| 32³ | xy, xz, yz | 4 | 337.15 ms | 48.51 ms | 6.95× |
| 64³ | xy, xz, yz | 2 | 281.56 ms | 154.91 ms | 1.82× |

Background load caused substantial timing variation; these are local loss
measurements, not whole-training-step or GPU speedups. JSON includes every sample,
min/max, environment and reference hash. The recorded local result is
`run/checks/anchor-neighborhood-20261013/benchmark.json`; `run/` is not committed.
Use `--device cuda` to measure a CUDA installation. Correctness tests retain the
loop reference and cover FP32/FP16/BF16, partial and overlapping anchors, all axes,
CFG dropout, fixed endpoints, and absent far references. CUDA autocast is tested
only when CUDA is available.

Training defaults are defined in [defaults.py](../src/config/defaults.py).
`normalize_train_config` fills active settings before assembly. The shared LR/SR
trainer receives inactive options through `get_trainer_sections`; those options
stay out of saved YAML, preserving the stage-specific schema. The builder reads
resolved values without per-key fallbacks, including explicit zero values.

[Back to README](../README.md)
