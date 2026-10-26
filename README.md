# Diffusion-GAN 3D

Generate 3D microstructures from **2D label images**, without measured 3D training targets.
Learn the appearance of real sections, guide generation with known planes,
and build larger or higher-resolution volumes.

[Install](#install) · [Run and inspect](docs/checks.md) · [Development](docs/development.md) · [Method & experiments](PAPER.md)

| 2D training image and example crops | Generated 3D volume |
| :---: | :---: |
| <img src="assets/paper/01-training-data.png" alt="Labeled 2D microstructure with example training crops" width="360"> | <img src="assets/paper/02-generated-volume.png" alt="Generated 3D microstructure with a cutaway view" width="320"> |

Examples from the [recorded experiments](PAPER.md); their settings differ from current defaults.

- **2D → 3D:** train from categorical sections in the `xy`, `xz`, and `yz` planes.
- **Plane anchors:** steer generation toward known sections by posterior sampling.
- **Larger outputs:** combine overlapping tiles; refine with a separate super-resolution model.

## Install

Use Python 3.11+. Install the [PyTorch build](https://pytorch.org/get-started/locally/) for
your GPU (CUDA or Apple MPS) or CPU first, then the rest:

```bash
git clone https://github.com/phykn/diffusion-gan3d.git
cd diffusion-gan3d
python -m venv .venv
source .venv/bin/activate   # PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train

Training has two stages. The **LR** (low-resolution) model generates volumes from noise
at `lo_res_size` voxels per `crop_size` source pixels (64 per 128 in the default sample).
The **SR** (super-resolution) model then refines LR volumes to `hi_res_size` voxels (128).

```bash
python scripts/01_check_dataset.py --sample default --no-view
python run_train.py default --run-dir "run/my-run"
python scripts/02_check_generated.py --weight "run/my-run-lr" --no-view
python scripts/06_check_hr.py --weight "run/my-run-sr" --no-view
```

`run_train.py` trains the LR model, then the SR model from frozen LR volumes and the real
high-resolution sections. `--run-dir NAME` writes `NAME-lr` and `NAME-sr`, and rerunning
the command skips a finished stage; without it, each stage gets a new timestamped folder in
`run/`. `--lr-only` stops after LR, so a later run with the same `--run-dir` trains only SR.
`--resume` continues one interrupted stage ([Save and resume](docs/checks.md#save-and-resume)).
The LR `generator.pt` is the best scored step, not the last one
([Best LR step](docs/checks.md#best-lr-step)).
The device is CUDA, then MPS, then CPU unless `--device` is given; mixed precision runs
in training on CUDA (on MPS only for the gradient-free sampling of each step), and in LR
and SR prediction on CUDA and MPS.

### Samples

A sample is one file in [config/samples/](config/samples/); copy
[default.yaml](config/samples/default.yaml) to add one.

- `images` maps `xy`, `xz`, and `yz` to folders of 2D uint8 label images with one shared
  pixel size; in `xz` and `yz` images, z runs down the rows. Labels are phases `0..K-1`,
  and phase 0 is pore. `K` is the largest label plus one unless `num_phases` is set.
- `resolution` sets the data size: each training crop is `crop_size` source pixels on
  an edge, sampled to `lo_res_size` voxels for LR and `hi_res_size` voxels for SR.
- `structure` says what the material looks like:
  - `isotropic: false` judges `xy` sections apart from `xz`/`yz` sections.
  - `depth_gradient: true` conditions both stages on height and, in LR, on the phase
    fractions at each height.
  - `mirror_symmetric: true` makes each volume's phase pairs along mirrored diagonals
    agree in LR (`loss.mirror_balance`), so the generator cannot shear the volume to keep two
    section planes realistic at the cost of the third. `false` turns that off and flips
    sections less often (probability 0.2), so a tilt stays visible.

  When z differs from x and y (either of the first two flags), training never rotates
  `xz`/`yz` sections, which would swap z with x or y; with a depth gradient it also
  never flips z.
- `tune.lr` and `tune.sr` change single settings of that stage.

Settings apply in order, later winning: code defaults, [config/base/](config/base/)`<lr|sr>.yaml`,
`structure`, then `tune`. `python run_train.py default --show` prints the YAML of both
stages without the code defaults.

## Web interface

Start the backend with an SR run, then the frontend in another terminal (Node.js 20.19+ or 22.12+):

```bash
python backend/run.py --weight "run/my-run-sr"
```

Each request generates the LR volume with the LR run the SR run was trained from
(`--lr-weight` picks another), then refines it; the viewer, porosity and tortuosity are
on the HR volume. Blocks count LR crops, so one block is `hi_res_size` voxels per edge.

```bash
cd frontend
npm ci
npm run dev
```

Open the URL Vite prints. For a non-default backend address, use
`API_URL=http://HOST:PORT npm run dev`; after `npm run build`, the backend serves the page
itself (default http://127.0.0.1:8000). The API is `GET /api/health`, `POST /api/prepare`,
and `POST /api/generate`; HTTP limits are in [backend/config.yaml](backend/config.yaml).

## Python API

[src/api.py](src/api.py) provides `LowResolutionAPI`, `PlaneAnchor`, `SuperResolutionAPI`,
and `extend_hr`. Volumes are `(z, y, x)`.

```python
from src.api import LowResolutionAPI, SuperResolutionAPI

lr = LowResolutionAPI("run/my-run-lr")  # a run directory or its generator.pt
labels = lr.generate(shape=(96, 64, 64), seed=1)  # uint8 phase labels
sr = SuperResolutionAPI("run/my-run-sr")
high = sr.refine(labels, seed=1, tile_size=64)  # (192, 128, 128) at the default 2x
```

`generate_probs` and `refine_probs` return phase fractions instead of labels and take the
same options; `seed=None` leaves the RNG unseeded.

Generated volumes are plausible samples, not measurements of an unknown 3D specimen.

[MIT License](LICENSE)
