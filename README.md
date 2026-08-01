# Diffusion GAN3D

Diffusion GAN3D learns categorical 3D microstructures from independent 2D slice datasets along three spatial axes. The slices need not be paired or aligned, and no ground-truth 3D volume is required.

## Method

The generator is a latent-conditioned 3D diffusion denoiser. Given `x_(t+1)`, transition `t`, and latent vector `z`, it predicts a clean categorical volume `x_0` that defines the reverse posterior step.

Three independent 2D critics evaluate slice pairs `(x_t, x_(t+1))`, one critic per axis. Real pairs come from forward-diffused training images; generated pairs come from the denoiser's reverse transition. Each critic combines a global head for whole-slice structure with a local head for phase boundaries.

Training alternates logistic critic updates with lazy R1 regularization and denoiser updates against all three critics. An exponential moving average of the denoiser is used for generation.

## Data

Training images are integer phase-label maps from `0` to `num_phases - 1`. The `data.folder.0`, `data.folder.1`, and `data.folder.2` directories contain slices normal to each axis and need not be aligned.

Labels are one-hot encoded and mapped to `[-1, 1]` during diffusion. The denoiser predicts one logit per phase, and final labels are selected with `argmax`.

`data.crop_size` is the size read from each independent 2D source image, `data.patch_size` is the 2D area seen by each critic, and `train.volume_sizes` lists the generated 3D training sizes. The source crop and generated volume sizes are independent. When an anchor image is smaller than a generated plane, it conditions only a partial plane without interpolation.

The included simulator creates an example three-phase dataset:

```bash
python gen_data.py
```

## Anchors

A plane anchor provides a known categorical slice at a selected axis and depth. During anchored training, one to `anchor.max_planes` real slices are placed at random axis/index positions and condition every reverse step. Axes may repeat, while parallel planes are kept apart by a count-dependent minimum gap. Anchor planes condition the prediction, cross-entropy trains the anchored logits, and focused critic samples cross an anchor plane instead of running parallel to it.

During sampling, anchors remain conditioning inputs throughout denoising; generated voxels are not overwritten.
All planes are merged into one condition, the model returns one global clean prediction, and one global posterior update follows. This uses the same global reverse-process lifecycle as scale-up; scale-up alone needs multiple tile predictions because its output is larger than the model input.
`anchor.dropout` is the probability that an otherwise available anchor condition is omitted for a training step.

## VF conditioning

Each training step computes one VF target from all pixels in the three real slice batches. The same target conditions every reverse transition, and an L1 loss compares it with the phase probabilities averaged over the complete generated 3D volume. Step-level condition dropout preserves an explicit unconditional generation path; the critics remain unconditional.
`vf.dropout` controls that step-level omission probability.

Generation accepts either that unconditional path or a manual phase vector:

```python
generator.generate(vf=None)
generator.generate(vf=(0.5, 0.1, 0.4))
```

Manual values must match `num_phases` and have a nonzero sum; `prepare_vf()` normalizes them to sum to one. Anchors and manual VF values can be used together.

## Scale-up

Scale-up performs joint tiled diffusion from one global noise volume. `data.patch_size` is the distance between adjacent tile starts and `overlap` is the one-side context, so each model input is `patch_size + 2 * overlap`. At every reverse transition, all tiles read the same unchanged state and share one latent vector. Their valid clean predictions are combined in global coordinates with cosine weights before each voxel receives one posterior update.

The two half-precision diffusion states remain on the GPU when they fit; otherwise they remain in CPU RAM and only the active tile is sent to the GPU. Generation returns one CPU `uint8` tensor and does not create output or temporary files. It accepts the same optional `vf` argument as regular generation and reuses one condition across every tile and reverse transition.

A supplied base keeps an inner region exact and uses a cosine transition across half of the selected overlap before meeting the generated surroundings. With zero overlap, the complete base remains exact.

## Training and outputs

Install the runtime dependencies and start training:

```bash
pip install -r requirements.txt
python run_train.py
```

Each run writes the EMA denoiser as `model.pt`, the three critics as `critic_0.pt` through `critic_2.pt`, the resolved `train.yaml`, and TensorBoard metrics under `run/<timestamp>/`. The four checkpoint paths load these weights independently; `null` starts the corresponding model from scratch. Optimizers, training step, and AMP scaler are not restored.

## Limitations

Matching 2D distributions along three axes constrains the generated microstructure but does not identify a unique 3D ground truth. Anchors provide local conditioning; the surrounding 3D continuation remains generated.

## References

- Johan Phan et al., “[Generating 3D images of material microstructures from a single 2D image: a denoising diffusion approach](https://www.nature.com/articles/s41598-024-56910-9),” *Scientific Reports* **14**, 6498 (2024).
- Zhisheng Xiao, Karsten Kreis, and Arash Vahdat, “[Tackling the Generative Learning Trilemma with Denoising Diffusion GANs](https://openreview.net/forum?id=JprM0p-q0Co),” *International Conference on Learning Representations* (2022).
- Omer Bar-Tal et al., “[MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation](https://arxiv.org/abs/2302.08113),” *International Conference on Machine Learning* (2023).
