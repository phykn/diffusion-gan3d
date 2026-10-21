# Diffusion-GAN 3D

Generate 3D microstructures from **2D label images**, without measured 3D training targets.
Learn the appearance of real sections, guide generation with known planes,
and build larger or higher-resolution volumes.

[Quick start](#quick-start) · [Run and inspect](docs/checks.md) · [Development](docs/development.md) · [Method & experiments](PAPER.md)

| 2D training image and example crops | Generated 3D volume |
| :---: | :---: |
| <img src="assets/paper/01-training-data.png" alt="Labeled 2D microstructure with example training crops" width="360"> | <img src="assets/paper/02-generated-volume.png" alt="Generated 3D microstructure with a cutaway view" width="320"> |

Examples from the [recorded experiments](PAPER.md); their settings differ from current defaults.

- **2D → 3D:** train from categorical sections in the `xy`, `xz`, and `yz` planes.
- **Plane conditions:** guide generation with known sections, called anchors.
- **Larger outputs:** combine overlapping tiles; refine with a separate super-resolution model.

## Quick start

Use Python 3.11+ with a PyTorch build for your GPU (CUDA or Apple MPS) or CPU.

```bash
git clone https://github.com/phykn/diffusion-gan3d.git
cd diffusion-gan3d
python -m venv .venv
source .venv/bin/activate   # PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Samples

Each sample is one file in [config/samples/](config/samples/), all with the shape of
[default.yaml](config/samples/default.yaml): `images` (a folder of 2D uint8 label images
per plane; the phase count comes from the images), `structure` (what the material looks
like), and `tune` (`lr`, `sr`: single values to change). Settings apply in order, later
winning: [config/base/](config/base/)`<stage>.yaml`, then `structure`, then `tune`;
`--show` prints the result.

```bash
python scripts/01_check_dataset.py --sample default --no-view
python run_train.py default --run-dir "run/my-run"
python scripts/02_check_generated.py --weight "run/my-run-lr" --no-view
python scripts/06_check_hr.py --weight "run/my-run-sr" --no-view
```

`run_train.py` trains the LR model, then the super-resolution (SR) model from its frozen
volumes and the real high-resolution sections. `--run-dir NAME` writes `NAME-lr` and
`NAME-sr`, and rerunning the command skips a finished stage; without it, each stage gets a
new timestamped folder in `run/`. `--show` prints the settings built for both stages, and
`--resume` continues an interrupted stage from its checkpoints. Inspection results go to
`checks/`. The device is CUDA, then MPS, then CPU unless `--device` is given; mixed
precision runs only on CUDA.

To add a sample, copy `default.yaml` and change its folders and `structure`.

## Web interface

Start the backend with an LR run, then the frontend in another terminal (Node.js required):

```bash
python backend/run.py --weight "run/my-run-lr"
```

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

[src/api.py](src/api.py) provides `LowResolutionAPI`, `PlaneAnchor`, and `SuperResolutionAPI`.
`generate` returns labels and `generate_probs` phase fractions, for example
`api.generate_probs(shape=(96, 64, 64), seed=1)`. SR's `refine` and `refine_probs` take an
LR volume and the same keyword options; `seed=None` leaves the RNG unseeded.

Generated volumes are plausible samples, not measurements of an unknown 3D specimen.

[MIT License](LICENSE)
