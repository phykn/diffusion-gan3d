# Diffusion GAN3D

Diffusion GAN3D generates categorical 3D microstructures from independent 2D slice datasets. It does not require paired slices or a ground-truth 3D volume.

The model learns whether a generated volume is plausible by comparing its 2D slices with real images from each spatial direction.

## Method

The generator is a latent-conditioned 3D diffusion denoiser. Given a noisy volume `x_(t+1)`, a transition index `t`, and a latent vector `z`, it predicts a clean categorical volume `x_0`.

The predicted clean volume is used in the diffusion posterior to sample the previous state:

```text
x_T → x_(T-1) → ... → x_1 → x_0
```

Training uses three independent 2D critics, one for each spatial axis. A critic receives a correlated slice pair `(x_t, x_(t+1))` and its transition index. Real pairs come from forward diffusion of training images; generated pairs come from one reverse transition of the 3D denoiser.

Each critic combines a deep global head for whole-slice structure with a shallow local head for fine phase boundaries. Their losses are averaged independently before the configured local weight is applied.

A training step therefore:

1. selects a random diffusion transition;
2. generates a 3D reverse-transition pair;
3. samples 2D pairs along axes `0`, `1`, and `2`;
4. updates each axis critic with logistic loss and lazy R1 regularization;
5. updates the denoiser against all three critics; and
6. updates an exponential moving average used for generation.

## Data representation

Each training image is a categorical label map:

```text
0, 1, ..., num_phases - 1
```

The folders configured as `data.folder.0`, `data.folder.1`, and `data.folder.2` contain slices normal to the corresponding spatial axis. Images from different folders do not need to be aligned.

Labels are converted to one-hot phase channels and mapped from `[0, 1]` to `[-1, 1]` before diffusion. The denoiser normalizes one logit channel per phase with softmax, maps the result back to `[-1, 1]`, and selects final labels with `argmax`.

The included simulator creates an example three-phase dataset:

```bash
python gen_data.py
```

## Plane anchors

An anchor provides a known categorical 2D plane at a selected axis and depth. It is encoded as one-hot phase channels plus a binary spatial mask and passed to the denoiser during every reverse transition.

During anchored training steps:

- one to three real training slices become orthogonal anchors;
- their axes, depths, and precedence at independent-data intersections are selected randomly;
- cross-entropy is applied to the predicted clean logits at all anchored voxels; and
- the ordinary three-axis adversarial objective still evaluates the volume.

During sampling, anchored voxels are projected into every reverse step. This keeps the constraint exact while allowing neighboring voxels to adapt throughout denoising.

Scale-up takes each new block's shared planes from the accumulated global phase consensus, then combines overlapping per-phase probabilities with complementary distance weights. Final categorical labels are selected only after this feathered overlap has been assembled; phase label numbers are never averaged directly.

Multi-axis scale-up should use weights trained with `anchor.max_planes: 3`; single-plane weights are accepted for inspection but produce a warning.

## Training

Install the runtime dependencies and start training:

```bash
pip install -r requirements.txt
python run_train.py
```

## Outputs

Each run stores the EMA denoiser as `model.pt`, the three critics as `critic_0.pt` through `critic_2.pt`, resolved training settings, and TensorBoard metrics under `run/<timestamp>/`. The four `train.checkpoint` paths initialize those models independently; a `null` entry starts that model from scratch. The step, optimizers, and AMP scaler always start fresh.

## Scope

Matching 2D distributions along three axes constrains a generated 3D microstructure, but it does not identify a unique 3D ground truth. Exact anchor projection preserves supplied planes, while the surrounding 3D continuation remains generated rather than uniquely determined.

## References

- Johan Phan et al., “[Generating 3D images of material microstructures from a single 2D image: a denoising diffusion approach](https://www.nature.com/articles/s41598-024-56910-9),” *Scientific Reports* **14**, 6498 (2024).
- Zhisheng Xiao, Karsten Kreis, and Arash Vahdat, “[Tackling the Generative Learning Trilemma with Denoising Diffusion GANs](https://openreview.net/forum?id=JprM0p-q0Co),” *International Conference on Learning Representations* (2022).
