# Diffusion-GAN 3D

Generate 3D microstructures from 2D label images, with optional anchor planes,
overlapping-tile generation, and a separate super-resolution stage.
Training uses real 2D sections without measured 3D targets.

## Setup

Use Python 3.11+ and a PyTorch build suitable for your CUDA environment.
Run commands from the project root in the activated project `.venv`.

```bash
git clone https://github.com/phykn/diffusion-gan3d.git
cd diffusion-gan3d
python -m venv .venv
# Activate .venv, then install dependencies:
python -m pip install -r requirements.txt
```

In PowerShell, use the interpreter directly without activation:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For development, install the test tools in the same environment. `httpx` is
required by the backend tests:

```powershell
& .\.venv\Scripts\python.exe -m pip install pytest ruff httpx
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m ruff check .
```

Frontend checks run from `frontend/`: `npm ci`, `npm test`, and `npm run build`.
Tests use a fixed available-RAM probe; dedicated memory tests supply their own
limits. CUDA tests are skipped when CUDA is unavailable.

## Train

Set image folders, phase counts, and resolutions in
[`config/data/default.yaml`](config/data/default.yaml). Inputs are 2D images
containing integer phase IDs (0–255, with `num_phases` between 1 and 256);
plane names are `xy`, `xz`, and `yz`.
Edit the [LR](config/train/low_res.yaml) and [SR](config/train/sr.yaml) training
presets as needed.

Datasets always return a dict containing `image` and source/crop metadata.
With `conditioning.height_enabled: true`, height is fixed to z: the vertical
direction of `xz`/`yz` sections. No height-axis setting is needed; side-image
heights are saved for inference. `xy` sections have no measured z coordinate
(`height_origin` and `height_extent` are -1).

```bash
python run_train_1st.py --device cuda
python run_train_2nd.py --base-weights "run/my-lr-run" --device cuda
```

SR trains from a frozen LR model and real HR sections. Weights, resolved settings,
and metrics are saved under `run/`.

## Generate and inspect

Pass a run directory or a `generator.pt` file:

```bash
python scripts/02_check_generated.py --weight "run/my-lr-run"
python scripts/03_check_anchor.py --weight "run/my-lr-run"
python scripts/04_check_scale_up.py --weight "run/my-lr-run"
python scripts/06_check_hr.py --weight "run/my-sr-run"
```

Results are saved under `run/checks/`. Add `--no-view` to save without opening a
viewer. See the [manual-check guide](docs/checks.md) for all six scripts.

Height-conditioned inference uses the saved side-image height and starts at
`height_origin=0` by default. If source heights differ within a domain, pass the
intended source's `height_extent`; use `height_origin` for an offset crop.
Both values use original image pixels, and the requested volume must fit inside
that height.

## Web interface

Start the backend, then run the frontend in another terminal (Node.js required):

```bash
python backend/run.py --weight "run/my-lr-run"
```

```bash
cd frontend
npm ci
npm run dev
```

Server limits are configured in [`backend/config.yaml`](backend/config.yaml).
Select the section plane and, for multi-domain models, the domain in the sidebar.
Height-conditioned xz/yz inputs use the crop row as their Z origin and the full
uploaded image height as their extent. XY inputs require separate Z coordinates.
Anchors start at the output origin, including when generating multiple blocks.

## Project layout

| Path | Purpose |
| --- | --- |
| `src/` | Models, data preparation, training, inference, and evaluation |
| `config/` | Data, training, and generation settings |
| `scripts/` | Numbered inspection scripts |
| `backend/`, `frontend/` | HTTP service and web UI |
| `simul/` | Synthetic data generation; run `python simul/run.py` |
| `run/` | Local weights and generated outputs |

[Method and recorded experiments](PAPER.md) · [MIT License](LICENSE)
