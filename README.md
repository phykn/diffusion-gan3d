# Diffusion-GAN 3D

Generate 3D microstructures from **2D label images**, without measured 3D training targets.
Learn the appearance of real sections, guide generation with known planes,
and build larger or higher-resolution volumes.

[Install](#install) · [Run and inspect](scripts/README.md) · [Method & experiments](paper/README.md)

| 2D training image and example crops | Generated LR volume (64³) |
| :---: | :---: |
| <img src="paper/assets/01-training-data.png" alt="Labeled 2D microstructure with example training crops" width="360"> | <img src="paper/assets/02-generated-volume.png" alt="Generated 64-cubed LR microstructure with a cutaway view" width="320"> |

The volume comes from the LR stage of the run described in [the paper](paper/README.md), with the
default sample's settings; one voxel spans two source pixels.

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
The **SR** (super-resolution) model then refines LR volumes `sr_scale` times finer
(64 to 128 voxels in the default sample).

```bash
python scripts/01_check_dataset.py --sample default --no-view
python run_train.py default --run-dir "run/my-run"
python scripts/02_check_generated.py --weight "run/my-run-lr" --no-view
python scripts/06_check_hr.py --weight "run/my-run-sr" --no-view
```

`run_train.py` trains LR, then SR from frozen LR volumes. `--run-dir NAME` writes `NAME-lr`
and `NAME-sr` and skips a finished stage on rerun; `--lr-only` stops after LR. Each stage's
`generator.pt` is its best scored step, not the last one. Resume, burn-in, devices, and
memory options are in [Training options](scripts/README.md#training-options).

### Samples

A sample is one file in [config/samples/](config/samples/); copy
[default.yaml](config/samples/default.yaml) to add one.

- `images` maps `xy`, `xz`, and `yz` to folders of 2D uint8 label images (phases `0..K-1`,
  phase 0 pore) with one shared pixel size; in `xz` and `yz`, z runs down the rows.
- `resolution`: training crops are `crop_size` source pixels on an edge, resized to
  `lo_res_size` voxels for LR and `lo_res_size × sr_scale` for SR.
- `structure`: `isotropic`, `depth_gradient`, and `mirror_symmetric` describe the material.
- `tune.lr` and `tune.sr` change single settings of a stage.

`python run_train.py default --show` prints the resolved YAML of both stages. Details of each
field are in [Training options](scripts/README.md#sample-fields).

## Web interface

Start the backend with an SR run, then the frontend in another terminal (Node.js 20.19+ or 22.12+):

```bash
python backend/run.py --weight "run/my-run-sr"
```

Each request generates an LR volume with the SR run's LR run (`--lr-weight` picks another)
and refines it.

```bash
cd frontend
npm ci
npm run dev
```

Open the URL Vite prints (`API_URL=http://HOST:PORT` for another backend); after
`npm run build`, the backend serves the page itself at http://127.0.0.1:8000.

## Python API

[src/api.py](src/api.py) provides `LowResolutionAPI`, `PlaneAnchor`, `SuperResolutionAPI`,
and `extend_hr`. Volumes are `(z, y, x)`.

```python
from src.api import LowResolutionAPI, SuperResolutionAPI

lr = LowResolutionAPI("run/my-run-lr")  # a run directory or its generator.pt
labels = lr.generate(shape=(96, 64, 64), seed=1)  # uint8 phase labels
sr = SuperResolutionAPI("run/my-run-sr")
high = sr.refine(labels, seed=1, tile_size=128)  # (192, 128, 128) at the default 2x
```

`generate_probs` and `refine_probs` return phase fractions instead of labels and take the
same options; `seed=None` leaves the RNG unseeded.

Generated volumes are plausible samples, not measurements of an unknown 3D specimen.

[MIT License](LICENSE)
