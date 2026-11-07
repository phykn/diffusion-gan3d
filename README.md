# Diffusion-GAN 3D

Generate 3D microstructures from **2D label images**, without measured 3D training targets.
Learn the appearance of real sections, guide generation with known planes,
and build larger or higher-resolution volumes.

[Install](#install) · [Run and inspect](scripts/README.md) · [Method & experiments](paper/README.md) · [Changes from v1](#changes-from-v1)

| 2D training image and example crops | Generated HR volume (128³) |
| :---: | :---: |
| <img src="paper/assets/01-training-data.png" alt="Labeled 2D microstructure with example training crops" width="360"> | <img src="paper/assets/sr-volume.png" alt="Generated 128-cubed HR microstructure with a cutaway view" width="320"> |

The volume is an LR sample refined by the SR stage of the run described in [the paper](paper/README.md),
trained with the default sample by v1 at commit f113ab8; one voxel spans one source pixel.

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
python scripts/check/01_dataset.py --sample default --no-view
python train.py default --run-dir "run/my-run"
python scripts/check/02_generate.py --weight "run/my-run-lr" --no-view
python scripts/check/04_hr.py --weight "run/my-run-sr" --no-view
```

`train.py` trains LR, then SR from frozen LR volumes. `--run-dir NAME` writes `NAME-lr`
and `NAME-sr` and skips a finished stage on rerun; without it, both stages go to
`run/<YYYYMMDD-HHMM>-<sample>-lr` and `-sr`. `--lr-only` (with `--run-dir`) stops after
LR. `--steps` and `--sr-steps` set the LR and SR totals (10000 and 5000 by default), and
`--bank-size` the frozen LR volumes that SR trains on (8). Each stage's `generator.pt` is
its best scored step, not the last one. Burn-in, devices, and memory options are in
[Training options](scripts/README.md#training-options); resume is in
[Save and resume](scripts/README.md#save-and-resume).

### Samples

A sample is one file in [config/samples/](config/samples/); copy
[default.yaml](config/samples/default.yaml) to add one. It names folders of 2D uint8 label
images for the `xy`, `xz`, and `yz` planes (phases `0..K-1`, phase 0 pore), the crop and
voxel sizes, what the material looks like, and single settings to change per stage. Each
field is described in [Sample fields](scripts/README.md#sample-fields).
`python train.py default --show` prints every setting of both stages.

## Web interface

Start the backend with an SR run, then the frontend in another terminal (Node.js 20.19+ or 22.12+):

```bash
python app/backend/run.py --weight "run/my-run-sr"
```

Each request generates an LR volume with the LR model stored in the SR run
(`--lr-weight` picks another LR export) and refines it.

```bash
cd app/frontend
npm ci
npm run dev
```

Open the URL Vite prints (`API_URL=http://HOST:PORT` for another backend). After
`npm run build`, a restarted backend serves the page itself at http://127.0.0.1:8000
(`--host` and `--port` change it).

## Python API

[src/api.py](src/api.py) provides `LowResolutionAPI`, `PlaneAnchor`, and
`SuperResolutionAPI`. Volumes are `(z, y, x)`.

```python
from src.api import LowResolutionAPI, SuperResolutionAPI

lr = LowResolutionAPI("run/my-run-lr")  # a run directory or its generator.pt
labels = lr.generate(shape=(96, 64, 64), seed=1)  # uint8 phase labels
sr = SuperResolutionAPI("run/my-run-sr")
high = sr.refine(labels, seed=1, tile_size=128)  # (192, 128, 128) at the default 2x
```

`generate` takes unset guidance, anchor strength, and overlap from
[config/gen.yaml](config/gen.yaml); `anchors=[PlaneAnchor(labels, axis, index)]` steers it
toward a known 2D section at `index` along `axis` (0, 1, 2 for `xy`, `xz`, `yz`).
Exactly one of `blocks`, `shape`, and `size` sets the output: `size=n` is `shape=(n, n, n)`,
and every output is tiled the same way, so the same seed and size give the same volume.
Guidance acts only on volume-fraction or depth-profile conditions; at guidance 1, or
without either condition, each step is a single forward pass.
`refine` refines in one block unless `tile_size` is given; `sr.default_tiling()` returns a
run's SR training size (128 for the default sample), and an unset `overlap` is
scale-aligned to the tile. `sr.lr` is the LR model stored in the SR export.
`generate_probs` and `refine_probs` return phase fractions instead of labels and take the
same options; `seed=None` leaves the RNG unseeded. `estimate_memory` and `check_memory`
check a request against free memory before it runs.

Generated volumes are plausible samples, not measurements of an unknown 3D specimen.

## Changes from v1

The code was rebuilt from v1 (commit 0608ab8, tag `v1`). Methods, defaults, and
TensorBoard tags are unchanged; the file formats below are not, and v2 reads none of the
v1 ones. Retrain with v2, or check out the `v1` tag to open a v1 run.

**Removed**

- Several domains (image sets) per model: the `--domain` script flag, the `domain` API and
  HTTP option, domain dropout, and SR guidance (`refine(guidance=…)`, `06_hr.py --guidance`).
- `extend_hr` and the `base`, `base_offset`, and `preserve_base` generation options, with
  the `--count`/`--gt` base demo of `04_scale_up.py`. Tiled generation covers large volumes.
- `data.split`, manual plane augmentation (`augmentation.planes`, `auto_planes`,
  `real_only`), `train.num_workers`, and `scripts/experiments/` with its options
  (critic `input_mode: single`, diffusion `time_embedding: scaled`).
- `config/base/`: code defaults live in `src/config/schema.py` only.

**Changed formats**

- Export (`generator.pt`): `{format: "diffusion-gan3d.export", version: 2, stage: lr|sr,
  step, config, model}`. An SR export also holds its LR export under `lr`, so moving or
  deleting the LR run no longer breaks SR; `--lr-weight` only overrides it.
- `train.yaml` is always the full resolved setting, code defaults included, for new and
  resumed runs alike. `stage` is `lr` or `sr`; `data.images.{xy,xz,yz}` replaces
  `data.domains.0`; `data.height_extent` and `data.depth_profile` replace the per-domain
  `height_extents` and `depth_profiles`; SR `source` is `{lr_run, lr_step}` instead of
  paths and hashes; `train.seed` is new; SR has no `loss.mirror_balance`.
- Run folder: no `lr_bank/`, `selection/inputs.pt`, or `critic_<group>.pt`; the SR bank,
  scoring inputs, and critics are inside `checkpoints/step_%08d.pt`. Without `--run-dir`
  a run is named `run/<YYYYMMDD-HHMM>-<sample>-lr|sr`.
- HTTP `GET /api/health`: `num_domains` is gone and `height_extent` (a number or null)
  replaces `height_extents`. `POST /api/generate` rejects a `domain` field with 422.

**Renamed check scripts**

| v1 | v2 |
| --- | --- |
| `02_generated.py`, `04_scale_up.py` | `02_generate.py` (`--blocks`, seams) |
| `03_anchor.py`, `05_continuation.py` | `03_anchor.py` (`--anchor PNG --side start\|end`) |
| `06_hr.py` | `04_hr.py` |
| `07_tilt.py` | `05_tilt.py` |

**Changed behavior**

- `--resume` always needs `--run-dir`; v1 named an SR resume itself. An SR resume needs
  nothing but the checkpoint.
- `size=n` means `shape=(n, n, n)` and is tiled; in v1 it was one untiled block of any size.
- New API options: `margin` (LR `generate` and `estimate_memory`), `anchors` and
  `guidance` in `estimate_memory`, `SuperResolutionAPI(…, lr_weights=…)` and `.lr`, and
  `storage` in `refine`.
- A resume continues the saved random state, so it gives the same losses as an
  uninterrupted run on CPU.
- With `lr_bank.flips: false`, SR no longer swaps y and x in its LR volumes, since that
  also reverses a tilt.
- The web app anchors the center of the first section instead of its corner and says
  what the anchor covers when there are several blocks.

[MIT License](LICENSE)
