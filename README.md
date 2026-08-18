# Anchor-Conditioned Diffusion for Scalable 3D Microstructure Synthesis

Generate natural 3D microstructures from 2D sections. The model supports:

- unconditional and domain-conditioned generation
- generation around a supplied 2D anchor
- large-volume generation with overlapping blocks

Unlike post-generation pasting, anchor conditioning is applied throughout
denoising so the surrounding 3D structure can adapt naturally. See
[`PAPER.md`](PAPER.md) for the method and evaluation.

## Install

```bash
git clone https://github.com/phykn/diffusion-gan3d.git
cd diffusion-gan3d
pip install -r requirements.txt
```

## Train

Set the 2D label-image folders and training options in
[`config/train.yaml`](config/train.yaml), then run:

```bash
python run_train.py --device cuda
```

## Generate

```python
from src.api import InferenceAPI, PlaneAnchor

api = InferenceAPI("run/my-experiment/generator.pt", device="cuda")
anchor = PlaneAnchor(image, axis=0, index=0)

direct = api.generate(seed=0)
anchored = api.generate(anchors=(anchor,), seed=0)
scaled = api.generate(anchors=(anchor,), blocks=(3, 3, 3), seed=0)
```

## Web interface

Build the Vue frontend once after cloning or changing files under `front/`:

```bash
cd front
npm ci
npm run build
cd ..
```

The generated `front/dist/` directory is intentionally not committed. Start the
API after the build completes:

```bash
python run_api.py --weight run/my-experiment/generator.pt --device cuda
```

Open <http://127.0.0.1:8000/> to crop an input section, generate a 3D volume,
inspect its phases, and reveal its continuation along axis 0.
PNG label sections are decoded from their raw values: indexed PNG palette
indices, grayscale PNG samples, or RGB PNGs whose three channels are identical.
No luminance conversion or palette-color mapping is applied, and values are
resized with nearest-neighbor sampling.

## Citation

```bibtex
@software{phykn2026anchorconditioneddiffusion,
  author = {phykn},
  title = {Anchor-Conditioned Diffusion for Scalable 3D Microstructure Synthesis},
  year = {2026},
  url = {https://github.com/phykn/diffusion-gan3d}
}
```
