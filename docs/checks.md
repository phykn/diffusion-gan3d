# Run and inspect

Run from the project root with the environment activated. `--weight` accepts a run
directory or its `generator.pt`. Each inspection saves results in `run/checks/`
and opens a viewer; add `--no-view` for files only.

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

`--blocks D H W` counts tiles along `(z, y, x)`. `--axis 0/1/2` selects `xy/xz/yz`.
Anchor demos use generated references by default; supply `--anchor` to script 05
for a measured boundary image. These comparisons do not establish measured 3D accuracy.

## Options and outputs

- `--device cpu`, `--seed`, `--domain`: device, repeatable sample, and dataset ID (`02–06`).
- `--napari`: open a 3D viewer (`02–06`). `--help` lists each script's full options.
- `--height-origin`, `--height-extent`: Z origin and full source height in **source pixels**.
  Height is the row direction in xz/yz images. Script 05 infers it from its input crop;
  set the extent explicitly when source heights differ.
- `01` saves a crop PNG; `02–05` save `volume.tiff`, `volume.png`, and `volume.json`.
  `06` saves LR/HR volumes, `comparison.png`, `report.json`, and optional LR fractions.

LR guidance, anchor strength, and overlap default to [config/gen.yaml](../config/gen.yaml).
Anchor strength is a 0–1 logit interpolation, not a pixel-match percentage.
Script 06 uses SR guidance `1.0` unless overridden. Its saved LR source is hash-checked;
`--lr-weight` selects an alternative and records it as unverified in `report.json`.

## Save and resume

`generator.pt` is for inference. Training checkpoints under `checkpoints/` include
optimizer state; keep their matching `.complete` files. `weights_every_steps`
overwrites exports, `archive_every_steps` creates checkpoints, and `structure_every_steps`
controls diagnostics. Completion and Ctrl+C at a completed step also save progress.
Load exports after writing has finished.

Resume requires the current checkpoint schema, including `path_maps` (an empty
list when no paths have moved). LR checkpoints also require the connectivity model,
optimizer, and update counter even when replay losses are disabled. Checkpoints
missing these fields are rejected; they are not upgraded during loading.

```bash
python run_train_1st.py --resume "run/my-lr-run" --steps 20000 --device cuda
```

`--steps` is the total target, including completed steps. Resume uses saved settings
and creates a new run directory. Both training commands accept `--path-map OLD NEW`
when files move; repeat it for separate data/run roots. Copy the original data and
completed artifacts. SR also needs its unchanged frozen LR weights/config and saved bank.

[Back to README](../README.md)
