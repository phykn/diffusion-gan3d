# Anchor-Conditioned Diffusion for Scalable 3D Microstructure Synthesis

Three-dimensional imaging is expensive, so material structures are often known only through a few measured 2D sections. Existing generators can produce plausible 3D volumes, but they do not preserve a section at its measured location. Pasting it into the result afterward can also create an artificial discontinuity.

This project uses measured sections as anchors throughout denoising, preserving their phases while generating a connected 3D structure around them. The anchored volume can then be extended through jointly generated overlapping blocks, allowing the measured structure to remain embedded in a larger microstructure.

See [`PAPER.md`](PAPER.md) for the complete method, experiments, and evaluation results.

## Installation

```bash
git clone https://github.com/phykn/diffusion-gan3d.git
cd diffusion-gan3d
pip install -r requirements.txt
```

## Usage

Set the 2D section folders and training options in [`config/train.yaml`](config/train.yaml), then train the model:

```bash
python run_train.py --device cuda
```

Generate a 3D volume around a known section and extend it to `3 × 3 × 3` blocks:

```python
from src.anchor import PlaneAnchor
from src.build import load_generator
from src.generate import ScaledGenerator

generator = load_generator(model_path, device)

anchor = PlaneAnchor(image, axis=0, index=32)
base = generator.generate(anchors=(anchor,))

volume = ScaledGenerator(generator).generate(
    blocks=(3, 3, 3),
    overlap=16,
    base=base,
)
```

## Citation

```bibtex
@software{phykn2026anchorconditioneddiffusion,
  author = {phykn},
  title = {Anchor-Conditioned Diffusion for Scalable 3D Microstructure Synthesis},
  year = {2026},
  url = {https://github.com/phykn/diffusion-gan3d}
}
```
