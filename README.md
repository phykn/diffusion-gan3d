# Diffusion GAN3D

Diffusion GAN3D learns categorical 3D microstructures from independent, unaligned 2D slice datasets along three axes.

## Idea

Real 3D training data is often unavailable, while representative 2D sections are comparatively easy to obtain. This project therefore generates a 3D volume but trains it through three 2D critics, each responsible for slices normal to one axis. A plausible result must match the measured material from every direction.

The generator uses a short diffusion process so that global structure can be refined over several transitions without the cost of a long conventional diffusion schedule. The critics evaluate both whole-slice structure and local phase boundaries.

## Data

`data.folder.0`, `data.folder.1`, and `data.folder.2` are lists of folders containing slices normal to each axis. Images must be `uint8` phase-label maps with values from `0` to `num_phases - 1`.

- `crop_size`: area cropped from each source image
- `patch_size`: resized 2D training size and scale-up tile spacing
- `augment`: randomly apply one of four rotations and their flipped variants
- `volume_sizes`: generated 3D sizes used during training

The three size settings are independent. Create an example three-phase dataset with:

```bash
python gen_data.py
```

## Training

```bash
pip install -r requirements.txt
python run_train.py
```

Each run saves the EMA generator as `model.pt`, the critics as `critic_0.pt` through `critic_2.pt`, the resolved `train.yaml`, and TensorBoard metrics under `run/<timestamp>/`. Checkpoint paths restore model weights only; optimizers, the training step, and the AMP scaler start fresh.

## Anchors

An anchor is a known categorical plane used throughout denoising. It is not copied into the result because exact overwriting can leave a discontinuity beside the plane. Cross-entropy teaches the plane itself, while seam-focused critic samples evaluate its surroundings. Each training step places all anchors on one random axis so independent crops cannot conflict at intersections. Training samples up to `anchor.max_planes`; `anchor.seam_weight` controls the seam loss and `anchor.dropout` preserves generation without anchors.

## VF conditioning

VF conditioning separates phase composition from spatial structure. Training derives a target from the real slice batches, while the critics still determine how those phases should be arranged. `vf.dropout` preserves generation without a VF target.

Manual VF values are normalized to sum to one:

```python
generator.generate(vf=(0.5, 0.1, 0.4))
```

Anchors and VF conditioning can be used together.

## Scale-up

Generating blocks independently creates seams because their boundary predictions are unrelated. Scale-up instead uses one global noise volume and combines overlapping clean predictions with cosine weights before each posterior update. Regions start `patch_size` voxels apart, and each model input has size `patch_size + 2 * overlap`.

Diffusion states remain on the GPU when they fit; otherwise they remain in CPU RAM and only the active tile is transferred. Generation returns a CPU `uint8` tensor and does not create output or temporary files.

An optional base expands an existing small volume. Its inner region remains exact while a cosine transition lets the generated surroundings connect to its boundary. With `overlap=0`, the complete base remains exact.

## Limitations

The method learns 3D consistency indirectly from 2D evidence. Matching all three slice distributions strongly constrains the result but cannot identify a unique ground-truth volume. Anchors constrain selected planes; everything between them remains a generated interpretation.

## References

- Johan Phan et al., “[Generating 3D images of material microstructures from a single 2D image: a denoising diffusion approach](https://www.nature.com/articles/s41598-024-56910-9),” *Scientific Reports* **14**, 6498 (2024).
- Zhisheng Xiao, Karsten Kreis, and Arash Vahdat, “[Tackling the Generative Learning Trilemma with Denoising Diffusion GANs](https://openreview.net/forum?id=JprM0p-q0Co),” *International Conference on Learning Representations* (2022).
- Omer Bar-Tal et al., “[MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation](https://arxiv.org/abs/2302.08113),” *International Conference on Machine Learning* (2023).
