# Run and inspect

Run from the project root. `--weight` accepts a run directory, its `generator.pt`,
or an EMA export under `checkpoints/`. Results go to a new folder under `checks/`;
add `--no-view` to skip the viewer in `01–06`.

| Script | Purpose | Weights |
| --- | --- | --- |
| [01_check_dataset.py](../scripts/01_check_dataset.py) | Preview training crops | None (`--sample`) or a run's data (`--weight`) |
| [02_check_generated.py](../scripts/02_check_generated.py) | Generate an LR volume | LR |
| [03_check_anchor.py](../scripts/03_check_anchor.py) | Compare anchored and unanchored samples | LR |
| [04_check_scale_up.py](../scripts/04_check_scale_up.py) | Tiled generation and seams | LR |
| [05_check_continuation.py](../scripts/05_check_continuation.py) | Grow a volume from a boundary section | LR |
| [06_check_hr.py](../scripts/06_check_hr.py) | Compare LR, upsampling, and SR | SR |
| [07_check_tilt.py](../scripts/07_check_tilt.py) | Signed particle tilt balance vs real volumes | LR (optional) |

```bash
python scripts/04_check_scale_up.py --weight "run/my-run-lr" --blocks 2 2 2 --no-view
python scripts/05_check_continuation.py --weight "run/my-run-lr" --anchor data/sample.png --axis 0
```

- `--blocks D H W` sets the `(z, y, x)` size to `patch + (n - 1) * (patch - 2 * overlap)`,
  where `patch` is the LR `lo_res_size` (default 64); the script prints the actual tile grid. `--axis 0/1/2` selects `xy/xz/yz`.
- `--device`, `--seed`, `--domain`, and `--napari` work in `02–06`; `--help` lists the rest.
- `--height-origin` and `--height-extent` (`02–07`) are in source pixels along the xz/yz row direction.
- `02 --vf-profile simul/config_anode.yaml --guidance 1.5` fixes the depth fractions to
  that simulation's targets (a profile YAML/JSON also works). Without it, a
  profile-conditioned model follows its training depth profile, since height alone
  flattens a strong gradient.
- `02–05` save `volume.tiff/png/json`; `06` saves LR/HR volumes, `comparison.png`, and `report.json`.
- `07` counts particles tilted each way (`dz/dy`, `dz/dx`, `dy/dx`); generated volumes
  should match `--real` (use `--real-stride 2` for 128-pixel crops at 64 voxels). Training
  logs the same balance as `structure/phase_N/tilt_balance_*` (pore phase 0 excluded);
  percolation is logged per transport axis as `structure/phase_N/percolation_z|y|x`, and
  section errors against real crops as `structure/<plane>/two_point_mae|chord_tv|cluster_error`.
- In `02–05`, guidance and overlap default to [config/gen.yaml](../config/gen.yaml), and so
  does the anchor strength of `03–05` (posterior-sampling step scale, 0 = off). In `06`,
  `--lr-guidance` uses that default,
  `--guidance` is the SR guidance (default 1.0), and overlap is scale-aligned. Anchor demos
  show controlled generation, not 3D accuracy.

## Save and resume

- `generator.pt` in the run directory is the inference export, refreshed every
  `weights_every_steps` and at the last step. In LR runs it is the best scored step so
  far, not the latest one ([Best LR step](#best-lr-step)); in SR runs it is the latest.
- Every `archive_every_steps` and at the last step, `checkpoints/` gets `step_N.pt` (full
  training state), `generator_N.pt` (EMA weights), and a `step_N.complete` marker that
  resume requires.
- The first Ctrl+C stops after the current step and saves both; a second stops at once
  without saving.
- Metrics and diagnostics go to `tensorboard/`; LR step scores also go to
  `selection/scores.yaml` ([Best LR step](#best-lr-step)).

```bash
python run_train.py --resume "run/my-run-lr" --steps 30000 --run-dir "run/my-run-2"
```

### Best LR step

LR training improves and then may degrade, for example when the critic memorizes few
real images. At each `weights_every_steps` save and the last step, LR training scores
its EMA weights: `selection.samples` unanchored volumes at guidance 1.0, from fixed
seeds whatever [config/gen.yaml](../config/gen.yaml) says, against 64 fixed real crops
per plane. Only 2D sections of the material exist, so the score compares sections:

- **Two-point** and **chord-length** errors, as in `structure/<plane>/*`.
- **Cluster error**: how far each phase's 4-connected clusters per section pixel and
  mean cluster area are from the real crops, relative to their mean. Two-point and chord
  statistics can match while pores merge into large blobs and the solid falls into
  fragments; clusters show what sections reveal of 3D connectivity.

Each error is divided by the same error between two halves of the real crops, so all
three use the sampling noise of the real data as their unit; **structure** is their
mean over planes. `generator.pt` is the step with the lowest structure. Height-conditioned
runs take each volume's depth from an `xz`/`yz` crop.

- `selection/scores.yaml` lists every scored step and the best step; TensorBoard has
  `selection/*`.
- `selection/generator_N.pt` keeps the best step's weights. Resume copies it from the
  resumed run, so keep it.
- One evaluation of 8 samples takes about 5 s per domain at 64³ on Apple MPS. Set
  `selection.enabled: false` to export the latest step instead.

Resume continues the stage the checkpoint belongs to with its saved settings, so it takes
no sample, `--sr-steps`, or `--bank-size`; `--steps` is that stage's total, including
completed steps. It writes a new run directory (here `run/my-run-2-lr`). After an LR
resume, `python run_train.py <sample> --run-dir "run/my-run-2"` reuses it and trains SR. Pass
`--path-map OLD NEW` (repeatable) when data or runs have moved.

[Back to README](../README.md)
