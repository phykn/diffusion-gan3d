# Manual inspection scripts

Run from the project root. Usually, only `--weight` is needed.
Both a `generator.pt` file and its run directory are accepted.

| Script | Inspect | Weights |
| --- | --- | --- |
| 01 | Crops from the training data | LR or SR; defaults to the current LR training preset if omitted |
| 02 | Basic LR generation | LR |
| 03 | The effect of multiple anchor planes | LR |
| 04 | Tiled generation and seams | LR |
| 05 | Continuation from a boundary plane | LR |
| 06 | LR, nearest-neighbor upsampling, and SR | SR; the LR source is read from the saved training settings |

```powershell
.venv\Scripts\python.exe scripts/01_check_dataset.py --weight "run/my-lr-run"
.venv\Scripts\python.exe scripts/02_check_generated.py --weight "run/my-lr-run"
.venv\Scripts\python.exe scripts/03_check_anchor.py --weight "run/my-lr-run"
.venv\Scripts\python.exe scripts/04_check_scale_up.py --weight "run/my-lr-run"
.venv\Scripts\python.exe scripts/05_check_continuation.py --weight "run/my-lr-run"
.venv\Scripts\python.exe scripts/06_check_hr.py --weight "run/my-sr-run"
```

Each run saves its results and opens a viewer. The output directory is printed
in the terminal: `run/checks/<timestamp>_<script-name>/`.

- `01`: training-crop preview PNG.
- `02–05`: `volume.tiff`, central xy/xz/yz previews in `volume.png`, and execution options in `volume.json`.
  `05` also saves a comparison of sections at increasing distances from the boundary.
- `06`: `lr.tiff`, `lr_probs.pt` when using phase fractions, `hr.tiff`, `comparison.png`, and `report.json`.

Add options only when needed:

| Option | Purpose |
| --- | --- |
| `--no-view` | Save without opening a viewer |
| `--napari` | Use a 3D viewer instead of 2D comparisons; scripts `02–06` |
| `--device cpu` | Run without a GPU; scripts `02–06` |
| `--seed 1` | Generate a different sample; default: `0` |
| `--domain 1` | Select another training domain; default: `0` |
| `--out PATH` | PNG for `01`, TIFF for `02–05`, output directory for `06` |
| `--help` | Show all options and an example command |

`03` and `05` take anchors from generated reference volumes, not measured 3D
truth. To use a real boundary image, add `--anchor image.png` to `05`.
`04` defaults to 2 × 2 × 2 tiles; use `--blocks D H W` to change the layout.

`02–05` read guidance defaults from `config/gen.yaml`. In `06`, SR guidance
defaults to `1.0` and LR guidance follows `config/gen.yaml`.
If the saved LR source has moved, add `--lr-weight "new/LR/path"` to `06`.

Shared helpers live in `scripts/common/`, experiment drivers in
`scripts/experiments/`, and paper reproduction tools in `scripts/paper/`.
Use the numbered scripts for routine inspection.
