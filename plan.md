# Diffusion GAN3D plan

## Data contract

Training reads only independent 2D categorical images from the three configured
axis folders. It must not read 3D TIFF volumes, bulk fractions, or statistics
derived from a 3D reference.

Each sample is a random `crop_size` crop followed by nearest-neighbor resizing
to `patch_size`.

## Method contract

- 3D time- and latent-conditioned U-Net predicts a clean categorical simplex.
- A variance-preserving diffusion posterior advances an 11-step reverse chain.
- Three independent time-conditioned 2D critics judge
  `(x[t-1], x[t], t)` pairs.
- Real pairs are produced by forward-noising real 2D crops.
- Fake pairs use exactly corresponding slices from a generated 3D transition.
- The reverse chain is built without gradients; one selected transition is
  recomputed with gradients.
- No phase-fraction or anchor conditioning is included in the baseline.

The source paper does not publish enough implementation detail for exact
reproduction. The schedule, loss, detach boundary, categorical representation,
and network sizes are therefore explicit implementation choices rather than
claims about unpublished author code.

## Gates

1. CPU shape, posterior, gradient, checkpoint, and data-leak tests.
2. 32-cube CUDA smoke run with a short schedule.
3. 64-cube, 11-step training with finite losses inside 6 GB VRAM.
4. Blind comparison of all three generated slice directions with independent
   real 2D crops.
5. Only after morphology and diversity pass may conditioning be considered.
