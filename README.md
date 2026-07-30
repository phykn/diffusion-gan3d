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

## Soft plane anchors

An anchor provides a known categorical 2D plane at a selected axis and depth. It is encoded as one-hot phase channels plus a binary spatial mask and passed to the denoiser during every reverse transition.

During anchored training steps:

- a real training slice becomes the anchor;
- its axis and depth are selected randomly;
- cross-entropy is applied to the predicted clean logits on that plane; and
- the ordinary three-axis adversarial objective still evaluates the volume.

The anchor is a learned condition, not a hard constraint. Its labels are never copied into the final output, so anchor accuracy must be evaluated after training.

## Training

Install the runtime dependencies and start training:

```bash
pip install -r requirements.txt
python run_train.py
```

## Outputs

Each run stores the EMA denoiser, three axis critics, resolved training settings, and TensorBoard metrics under `run/<timestamp>/`. Optimizer state is not saved, so training cannot be resumed exactly.

## Scope

Matching 2D distributions along three axes constrains a generated 3D microstructure, but it does not identify a unique 3D ground truth. Likewise, a soft anchor encourages consistency with a supplied plane but does not guarantee exact reproduction.

## References

- Johan Phan et al., “[Generating 3D images of material microstructures from a single 2D image: a denoising diffusion approach](https://www.nature.com/articles/s41598-024-56910-9),” *Scientific Reports* **14**, 6498 (2024).
- Zhisheng Xiao, Karsten Kreis, and Arash Vahdat, “[Tackling the Generative Learning Trilemma with Denoising Diffusion GANs](https://openreview.net/forum?id=JprM0p-q0Co),” *International Conference on Learning Representations* (2022).
