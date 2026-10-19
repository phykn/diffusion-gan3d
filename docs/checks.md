# Run and inspect

Run from the project root. `--weight` accepts a run directory, its `generator.pt`,
or an EMA export under `checkpoints/`. Results go to a new folder under `checks/`;
add `--no-view` to skip the viewer in `01–06`.

| Script | Purpose | Weights |
| --- | --- | --- |
| [01_check_dataset.py](../scripts/01_check_dataset.py) | Preview training crops | Optional (LR preset) |
| [02_check_generated.py](../scripts/02_check_generated.py) | Generate an LR volume | LR |
| [03_check_anchor.py](../scripts/03_check_anchor.py) | Compare anchor conditions | LR |
| [04_check_scale_up.py](../scripts/04_check_scale_up.py) | Tiled generation and seams | LR |
| [05_check_continuation.py](../scripts/05_check_continuation.py) | Grow a volume from a boundary section | LR |
| [06_check_hr.py](../scripts/06_check_hr.py) | Compare LR, upsampling, and SR | SR |
| [07_check_tilt.py](../scripts/07_check_tilt.py) | Signed particle tilt balance vs real volumes | LR (optional) |

```bash
python scripts/04_check_scale_up.py --weight "run/my-lr-run" --blocks 2 2 2 --no-view
python scripts/05_check_continuation.py --weight "run/my-lr-run" --anchor image.png --axis 0
```

- `--blocks D H W` sets the `(z, y, x)` size to `patch + (n - 1) * (patch - 2 * overlap)`;
  the script prints the actual tile grid. `--axis 0/1/2` selects `xy/xz/yz`.
- `--device`, `--seed`, `--domain`, and `--napari` work in `02–06`; `--help` lists the rest.
- `--height-origin` and `--height-extent` are in source pixels along the xz/yz row direction.
- `02 --vf-profile config/profile/simul_anode.yaml --guidance 1.5` fixes the depth
  fractions; height alone flattens a strong gradient.
- `02–05` save `volume.tiff/png/json`; `06` saves LR/HR volumes, `comparison.png`, and `report.json`.
- `07` counts particles tilted each way (`dz/dy`, `dz/dx`, `dy/dx`); generated volumes
  should match `--real` (use `--real-stride 2` for 128-pixel crops at 64 voxels). Training
  logs the same balance as `structure/phase_N/tilt_balance_*`.
- In `02–05`, guidance, anchor strength (0–1 logit interpolation), and overlap default to
  [config/gen.yaml](../config/gen.yaml). In `06`, `--lr-guidance` uses that default,
  `--guidance` is the SR guidance (default 1.0), and overlap is scale-aligned. Anchor demos
  show controlled generation, not 3D accuracy.

## Save and resume

- `generator.pt` in the run directory is the inference export, refreshed every
  `weights_every_steps`.
- Every `archive_every_steps`, `checkpoints/` gets `step_N.pt` (full training state),
  `generator_N.pt` (EMA weights), and a `step_N.complete` marker that resume requires.
- Metrics and diagnostics go only to `tensorboard/`.
- The first Ctrl+C stops after the current step; a second stops at once without saving.

```bash
python run_train_1st.py --resume "run/my-lr-run" --steps 20000
```

`--steps` is the total, including completed steps. Resume uses saved settings, writes a
new run directory, and rejects checkpoints from older schemas. Pass `--path-map OLD NEW`
(repeatable) when data or runs have moved.

[Back to README](../README.md)
