<div align="center">
  <h1>Anchor-Conditioned Diffusion for Scalable 3D Microstructure Synthesis</h1>
</div>

## Abstract

Three-dimensional microstructures must be large enough to represent connectivity and transport, yet volumetric imaging is often more costly than acquiring 2D sections. Matching 2D section distributions alone does not ensure that a specified section appears at a prescribed coordinate, and independently generated blocks can introduce discontinuities when a volume is enlarged. We address these problems with anchor-conditioned diffusion. Rectangular categorical section regions and their coordinates condition the denoiser, constrained voxels receive direct phase supervision, and adversarial and transition-distribution losses regularize the anchor neighborhood against triplets from unconditioned model samples. Anchors remain learned conditions throughout direct 128³ sampling; generated labels are neither clamped nor overwritten. Across four seeds, 25% and 100% axis-0 coverage produced 94.88 ± 0.03% and 97.93 ± 0.08% whole-volume agreement, respectively, with an in-distribution synthetic reference. A 3 × 3 × 3 fixed-block expansion generated a 352³ volume—20.80 times the base voxel count—with a mean local pore-continuation drop of 0.32 ± 0.22 percentage points at block boundaries. In a five-seed ablation, the exact-boundary phase-change ratio was 6.90 ± 0.52 without overlap and 0.99 ± 0.16 with eight-voxel inward margins. The evaluation uses one binary training image and a same-model synthetic reference.


## 1. Introduction

Transport and mechanical response depend on three-dimensional morphology, not only on the appearance of individual sections. Digital material studies therefore require volumes that are large enough to contain representative connected paths. Tomography can provide such data, but its cost, resolution, and field of view often make 2D microscopy the more accessible source of microstructural information.

Optimization, adversarial, and diffusion methods can infer statistically plausible 3D structures from one or more 2D images [1–5]. Their usual objective is distributional: generated sections should resemble the observed 2D population. A distributional objective by itself does not guarantee that a particular section is retained at a known coordinate. Inserting that section after generation satisfies the local labels but can break structures on either side of the plane.

Volume size creates a second difficulty. A model trained on small patches cannot directly represent a much larger field of view, whereas generating blocks independently and stitching them afterward leaves no mechanism for reconciling their boundaries. Shared-state diffusion offers a way to couple overlapping predictions during sampling [6].

We combine these ideas in a categorical 3D generator trained without paired 2D–3D data. The method (i) integrates full or rectangular partial plane anchors throughout denoising, with direct label supervision and adversarial and transition-distribution regularization; (ii) trains and applies overlapping tiled contexts around a softly conditioned base; and (iii) evaluates coordinate agreement, section distributions, phase fraction, diffusive tortuosity, three-axis pore percolation, and tile-boundary continuity separately. This separation is important because a volume can match 2D statistics without matching a controlled target at the same coordinates, or satisfy supplied coordinates while changing global morphology.

## 2. Related Work

### 2.1 Reconstruction from 2D observations

Feature-matching and conditional neural methods reconstruct 3D microstructures from a 2D exemplar [1,2]. SliceGAN removes the need for paired 3D supervision by applying 2D discriminators to sections of a generated volume [3]. Diffusion has also been used for 2D microstructure synthesis [7] and, more recently, for 2D-to-3D dimensional expansion [4,5]. Property-conditioned approaches control quantities such as phase fraction or spatial statistics [8,9]. Recent volumetrically supervised methods additionally condition on fixed or sparse observed slices [10,11]. Our distinct setting uses unregistered 2D section collections rather than volumetric training targets and treats supplied internal sections as learned conditions in direct, base-size samples.

### 2.2 Conditioning on known regions

Diffusion inpainting preserves observed pixels while generating their surroundings [12,13]. A specified internal section poses a stricter 3D problem: unknown material lies on both sides, and the generated neighborhood should remain compatible across the plane. We adapt masked conditioning to categorical internal sections and regularize three-plane stacks spanning each anchor; direct anchor-neighborhood evaluation remains future work.

### 2.3 Generation beyond the training size

MultiDiffusion binds overlapping diffusion paths through a shared optimization objective [6], whereas Patch-DM collages features cropped from neighboring patches [14]. GrainPaint applies diffusion inpainting to large microstructures [15]. Our scale-up procedure fuses overlapping clean predictions in one 3D categorical state and adds an adaptive base condition with a cosine-tapered transition shell.

## 3. Method

### 3.1 Task definition

Let $\mathcal{D}_a$ be a set of categorical 2D sections normal to axis $a \in \{0,1,2\}$, with phase labels in $\{0,\ldots,K-1\}$. The axis-specific sets need not be spatially aligned. We learn a generator whose categorical samples

$$
X \in \{0,\ldots,K-1\}^{D \times H \times W}
$$

have sections that match the corresponding 2D distributions. At inference, each anchor supplies either a full plane or one dense rectangular region at a distinct axis/coordinate pair. Multiple planes and axes are supported, but labels at intersecting planes must agree. Voxels outside these regions are sampled stochastically.

### 3.2 2D-supervised 3D diffusion

The generator contains a 3D denoising network $G_\theta(x_{t+1},t,z_t,c)$. Reverse transition $t$ maps noisy state $x_{t+1}$ to $x_t$; the network receives that current state, a per-transition latent vector $z_t$, and optional conditions $c$, and predicts phase logits $\ell_\theta$. These are decoded to the relaxed signed one-hot estimate $\hat{x}_0=2\,\mathrm{softmax}(\ell_\theta)-1$; categorical labels are obtained by a final phase-wise argmax. The model follows the forward–reverse diffusion formulation [16] and the short adversarial reverse process of denoising diffusion GANs [17]. The denoiser uses channel widths 16, 32, 64, and 64, a 128-dimensional conditioning embedding, and a 64-dimensional latent vector that is resampled at every reverse transition.

Because paired 3D targets are unavailable, three 2D critics $C_a$ supervise orthogonal sections. Each critic distinguishes forward-noised real section pairs from pairs sliced from the generated reverse process. A global head evaluates the whole section, and a patch head evaluates fine-scale structure. The generator objective is

$$
\mathcal{L}_{\mathrm{adv}}
=\sum_{a=0}^{2}\left(
\mathcal{L}_{\mathrm{global}}^{(a)}
+\lambda_{\mathrm{local}}\mathcal{L}_{\mathrm{local}}^{(a)}
\right).
$$

This trains a 3D generator through observable 2D distributions; it does not provide measured volumetric supervision.

### 3.3 Plane-anchor conditioning

An anchor contains a dense rectangular array of categorical labels, a plane normal, a coordinate on that axis, and an optional in-plane offset. Anchors at distinct axis/coordinate pairs are assembled into a target tensor $Y$ and binary mask $M$, where $M(v)=1$ marks constrained voxel $v$. Full planes and smaller rectangles use the same representation; intersecting rectangles must assign the same label to their shared voxels.

Define the anchor condition as $c_A=(\mathrm{onehot}(Y)\odot M,M)$. Its masked one-hot labels and mask are projected by a zero-initialized 3D convolution and added to the first denoiser feature map:

$$
h_A=\mathrm{Conv}_{3D}\!\left(
\left[\mathrm{onehot}(Y)\odot M,\;M\right]
\right).
$$

Zero initialization leaves the unconditioned mapping unchanged at the start of training. The anchor is provided at every reverse step, and constrained labels are optimized with masked cross-entropy,

$$
\mathcal{L}_{\mathrm{anchor}}
=-\frac{1}{|M|}\sum_{v:M(v)=1}
\log p_\theta\!\left(Y(v)\mid x_{t+1},t,z_t,c_A\right).
$$

Correct labels alone do not ensure a compatible neighborhood. At the final transition ($t=0$), a separate critic evaluates three-plane stacks that span an anchor along its normal direction and supplies $\mathcal{L}_{\mathrm{conn}}$. Its reference stacks are axis-matched triplets replayed from unconditional generated volumes, so the loss regularizes an anchored neighborhood toward the model's baseline continuation rather than toward measured 3D connectivity.

A separate teacher-volume bank stores volumes generated from real single-plane anchors. After the anchor ramp, a multi-plane teacher condition may sample several mutually registered planes from one stored volume, subject to a density limit, minimum same-axis spacing, and optional mixed axes. Keeping this bank separate from the triplet replay avoids treating unrelated real 2D sections as registered slices of one volume.

At inference, $c_A$ is supplied to the denoiser at every reverse transition with a user-controlled strength. Sampling starts from the same ordinary noise state as unconditioned generation, applies the standard reverse posterior, and returns a phase-wise argmax after the final transition. The sampler does not initialize constrained voxels from $q(x_T\mid Y)$, clamp intermediate clean predictions, or overwrite final labels. Anchor agreement is therefore a measured learned-conditioning outcome rather than a sampler invariant.

### 3.4 Phase-fraction conditioning and training objective

An optional phase-fraction vector $v\in\Delta^{K-1}$, with $v_k\geq0$ and $\sum_k v_k=1$, specifies the desired composition. Non-negative user inputs are normalized to this simplex before sampling. Its embedding conditions the denoiser, while predicted mean fractions $\hat{p}$ receive

$$
\mathcal{L}_{\mathrm{vf}}=\frac{1}{2}\sum_{k=0}^{K-1}|\hat{p}_k-v_k|.
$$

At training step $s$, the implemented generator objective is

$$
\mathcal{L}_{G}=\mathcal{L}_{\mathrm{adv}}
+r_A(s)\left(
\lambda_{\mathrm{anchor}}\mathcal{L}_{\mathrm{anchor}}
+\lambda_{\mathrm{conn}}\mathcal{L}_{\mathrm{conn}}
+\lambda_{\mathrm{normal}}\mathcal{L}_{\mathrm{normal}}
\right)
+\lambda_{\mathrm{vf}}\mathcal{L}_{\mathrm{vf}}
$$

where unavailable conditional terms are zero. The anchor ramp $r_A$ starts at step 3,000 and reaches one after 6,000 additional steps. The connectivity critic term is active only for anchored final-transition samples. On the same matched anchor triplets, $\mathcal{L}_{\mathrm{normal}}$ compares soft center-to-previous and center-to-next $K\times K$ phase-transition matrices with total-variation distance. The implementation uses 10 reverse transitions, $\lambda_{\mathrm{local}}=0.5$ inside $\mathcal{L}_{\mathrm{adv}}$, and weights $\lambda_{\mathrm{anchor}}=1$, $\lambda_{\mathrm{conn}}=0.25$, $\lambda_{\mathrm{normal}}=0.10$, and $\lambda_{\mathrm{vf}}=1$.

Anchor conditions are requested for 80% of eligible training samples. When anchor and phase-fraction conditions are both available, the joint-null, anchor-null-only, and phase-fraction-null-only states are mutually exclusive and each occurs with probability 0.05; both conditions are retained for the remaining 0.85. When no anchor is present, the lone phase-fraction condition is dropped with probability 0.10. This supplies unconditional and partially conditional states to the same network without independent 0.2 dropouts.

### 3.5 Fixed-block shared-state tiled scale-up

Let $P=128$ be the fixed block input size and $o$ the margin reserved inside each block on every shared face. Adjacent blocks start $P-2o$ voxels apart, so their predictions share a $2o$-voxel fusion band. For $b$ blocks, the output length is $P+(b-1)(P-2o)$. Every denoiser call therefore uses the same $P^3$ shape seen during training. At every reverse transition, one newly sampled latent vector $z_t$ is shared by all blocks. A separable cosine-taper window $w_k$ then fuses their overlapping predictions:

$$
\bar{x}_0(v)=
\frac{\sum_k w_k(v)\hat{x}_{0,k}(v)}
{\sum_k w_k(v)}.
$$

The reverse posterior updates the shared global state only after fusion, so adjacent blocks exchange information throughout denoising rather than being stitched after generation. Margins exist only on faces shared by blocks. Outermost faces use unit weight and receive no padding, external halo, or periodic wrap.

When a base volume $B$ is supplied, define its signed one-hot field as $b(v)=2\,\mathrm{onehot}(B(v))-1$ and keep one fixed noise field $\epsilon_B$. The corresponding fixed-noise forward realization is

$$
\widetilde{B}_{t+1}(v)=
\sqrt{\bar{\alpha}_{t+1}}\,b(v)
+\sqrt{1-\bar{\alpha}_{t+1}}\,\epsilon_B(v).
$$

Immediately before reverse transition $t$, the centered shared state is blended as

$$
x_{t+1}(v)\leftarrow[1-s(v)]x_{t+1}(v)
+s(v)\widetilde{B}_{t+1}(v),
$$

where $\bar{\alpha}_{t+1}$ is the cumulative forward signal coefficient and $s(v)=1$ in the base interior, with a separable cosine taper over the four-voxel outer shell. This operation conditions the noisy input state; it does not clamp the tile predictions. Final labels are obtained from the fused clean prediction and are never overwritten by $B$, so even the full-strength base interior may change.

## 4. Experimental Setup

### 4.1 Data and scope

The available source artifact, `data/sample.png`, is a 226 × 690 binary phase map. Phase 0 is treated as pore and phase 1 as solid. The material system, acquisition and segmentation procedures, physical pixel size, and external source or data license are not recorded in the available project files. Results are therefore reported in voxels and should be interpreted as an algorithmic study without a calibrated physical length scale. Under an isotropy assumption, the same image distribution supervises all three axes. Random 128 × 128 crops are used at the real critic-section size and, with probability 0.5, augmented by right-angle rotations and reflections (Figure 1).

<p align="center">
  <img src="assets/paper/01-training-data.png" alt="Binary training image with three 128 by 128 crop regions" width="680">
</p>
<p align="center"><em>Figure 1. Binary 2D training image. Orange boxes show example 128 × 128 crops used at the real critic-section size. Black denotes pore and gray denotes solid throughout.</em></p>

Evaluation uses 64 randomly sampled real crops. A fixed unconditioned 128³ seed-10000 sample from the trained generator serves as synthetic pseudo-ground truth for controlled anchor tests; it is denoted GT only in that context. Evaluation volumes use separate seeds 0–3. All real crops come from the training image, and no held-out experimental 3D volume is included.

### 4.2 Training and implementation

The evaluated model is the EMA output of a 20,000-step run with decay 0.999. The same checkpoint is retained for all results so the fixed-block experiment isolates the inference geometry change.

The recorded run sampled the relaxed 3D side length uniformly from 128 or 144 voxels; 16 section pairs per axis were passed to the critics. The real-section loader used batch size 8. Adam used learning rates $1.6\times10^{-4}$ for the denoiser and $1.0\times10^{-4}$ for all critics, with $\beta_1=0.5$ and $\beta_2=0.9$. Training used mixed precision and an R1 penalty with $\gamma=0.2$ every 16 steps. The 10-transition variance-preserving schedule used continuous rate endpoints $\beta_{\min}=0.1$ and $\beta_{\max}=20$, with $\bar{\alpha}(u)=\exp[-\tfrac{1}{2}(\beta_{\max}-\beta_{\min})u^2-\beta_{\min}u]$ for $u\in[0,1]$. Anchor conditioning started at step 3,000 and ramped over 6,000 steps; an anchor was requested on 80% of eligible steps. With both anchor and phase-fraction conditions available, joint-null, anchor-null-only, and phase-fraction-null-only states were mutually exclusive and each had probability 0.05; the remaining 0.85 retained both. When only phase-fraction conditioning was available, it was dropped with probability 0.10. After the anchor ramp, a multi-plane teacher was requested with probability 0.5; maximum plane density was 0.05, minimum same-axis spacing was four voxels, and mixed-axis probability was 0.5. The checkpoint also reflects the overlap-consistency auxiliary configured for that run; the fixed-block sampler itself is applied without retraining.

### 4.3 Evaluation protocols

The single-plane examples in Figures 4 and 6 use the 128 × 128 crop at $(\mathrm{left},\mathrm{top})=(281,58)$ in `data/sample.png`, place it at the axis-0 center of a 128³ volume, and use seed 0. A separate seed-0 coverage sweep supplies 0, 1, 2, 4, 8, 16, 32, 64, or 128 planes from the synthetic GT at evenly distributed axis-0 coordinates. Coordinates are recomputed for each count, so successive sets are not generally nested. The multi-sample evaluation uses 32, 64, 96, and 128 planes—25%, 50%, 75%, and 100% coverage—with seeds 0–3.

The phase-fraction-conditioned samples receive the synthetic reference fractions $(0.3390789,0.6609211)$ as an oracle target. Only this one target is tested.

The scale-up evaluation uses 3 × 3 × 3 fixed 128³ blocks with eight-voxel inward margins, producing a 352³ output with seams at coordinates 120 and 232 on each axis. A separate unanchored ablation fixes the block count at 2 × 1 × 1 and varies the margin over 0, 4, 8, 12, and 16 voxels. The corresponding axis-0 output lengths are 256, 248, 240, 232, and 224, with one seam at their respective midpoints; both transverse axes remain 128.

### 4.4 Metrics

- **Kernel Inception Distance (KID):** the unbiased squared maximum mean discrepancy between 2,048-dimensional Inception-v3 features [18], using 100 subsets of 50 images. Table 1 compares a fixed set of 64 real crops (seed 10000) with either an independently sampled real crop set or 64 axis-0 sections drawn without replacement from each generated volume. Figure 5 instead compares all 128 generated axis-0 sections with all 128 synthetic-GT sections. All inputs use a 128 × 128 field of view and are repeated into three binary RGB channels. For scale-up, one seeded 128 × 128 crop is taken from each selected section. Lower is better, and the unbiased estimate may be slightly negative.

- **Fréchet Inception Distance (FID):** the Fréchet distance between Gaussian fits to 2,048-dimensional Inception-v3 features [20]. Figure 5 compares all 128 generated axis-0 sections with all 128 synthetic-GT sections after the same binary-RGB conversion used for KID. Lower is better. This fixed-seed diagnostic is not reported in Table 1.

- **Porosity:** the fraction of pixels or voxels assigned to phase-0 pore. Agreement with the reference value is desired.

- **Tortuosity:** the axis-0 diffusive tortuosity factor of the pore phase, computed with TauFactor 1.2.1 using Dirichlet conditions along axis 0, no-flux transverse boundaries, and convergence criterion $10^{-3}$ [19]. It follows $D_{\mathrm{eff}}=D\varepsilon/\tau$, where $\varepsilon$ is porosity. Comparison with the synthetic reference measures within-model consistency, not agreement with experimental transport.

- **Voxel accuracy:** the fraction of all 128³ voxels whose phase matches the synthetic GT at the same coordinate. This is a whole-volume recovery score, not accuracy restricted to supplied planes.

- **Pore-percolation error:** phase 0 is the pore phase, and connected components use non-periodic 6-connectivity, so only face-sharing voxels are adjacent and opposite boundaries do not wrap. For axis $a$, let $S_a$ contain every phase-0 component that touches both opposing faces normal to $a$. Its percolating fraction is $P_a=\lvert\bigcup_{C\in S_a}C\rvert/\lvert\{v:X(v)=0\}\rvert$, where all spanning components are included. Figure 5 reports $E_{\mathrm{pore}}=\frac{100}{3}\sum_{a=0}^{2}\lvert P_a^{\mathrm{pred}}-P_a^{\mathrm{GT}}\rvert$ in percentage points. Lower is better. If either volume has no phase-0 voxels, the metric is undefined and evaluation raises an error.

- **Local pore-continuation drop:** for adjacent planes normal to axis $a$, $C_a=P(X_{i+1}=0\mid X_i=0)$. The reported value is the three-axis mean $\Delta C=C_{\mathrm{interior}}-C_{\mathrm{boundary}}$, excluding pairs within four voxels of each boundary from the interior estimate. Zero indicates no measured boundary effect; a negative value means slightly greater local continuation at the boundary.

- **Overlap diagnostics:** the exact seam-change ratio divides the phase-change rate at the tile boundary by the median of the remaining pair rates; the evaluator's radius-4 exclusion removes the boundary pair together with its four lower-index and five higher-index neighbors. A ratio of one is ideal. For the other diagnostics, the seam-band half-width is $b=\max[1,\min(o,4)]$. Each adjacent-plane pair in that band is compared with a pooled interior distribution, subsampled to at most 64 pairs. Transition TV is the maximum total-variation distance between phase-transition distributions, and continuation delta is the maximum absolute phase-wise continuation difference; lower is better for both.

Unless marked otherwise, Table 1 values are means over four seeded replicates. KID ± is the mean of the four within-replicate sample standard deviations across 100 KID subsets; every other ± value is the between-replicate sample standard deviation. Depending on the row, replicate variation arises from volume sampling, crop selection, and/or section selection. These uncertainties do not cover specimens, datasets, or training runs. The controlled reference is one volume. The real-data KID is a baseline between independently sampled crop sets. Figure 5 is a separate seed-0 diagnostic: KID and FID are measured against synthetic-GT sections, and pore-percolation error compares the generated and GT phase-0 spanning-component fractions over all three axes. No across-volume uncertainty is shown for this fixed-seed sweep.

## 5. Results

Table 1 reports KID, porosity, tortuosity, voxel accuracy, and local pore-continuation drop. Real-versus-real KID is 0.0004 ± 0.0013 and unconditioned 3D KID is 0.0146 ± 0.0016. The synthetic reference has porosity 0.3391 and axis-0 tortuosity 2.1025; the unconditioned means are 0.3430 and 2.0444. With the reference phase fractions supplied as the single oracle target, the corresponding values are KID 0.0414, porosity 0.3530, and tortuosity 1.9173. The fixed-block scale-up has KID 0.0091 ± 0.0014, porosity 0.3400 ± 0.0029, and tortuosity 2.0967 ± 0.0154.

| Evaluation data | KID | Porosity | Tortuosity | Voxel accuracy | Local pore-continuation drop |
|---|---:|---:|---:|---:|---:|
| Controlled reference (GT) | — | 0.3391 | 2.1025 | — | — |
| Real 2D crops | 0.0004 ± 0.0013 | 0.3625 ± 0.0042 | — | — | — |
| 3D | 0.0146 ± 0.0016 | 0.3430 ± 0.0039 | 2.0444 ± 0.0928 | — | — |
| 3D (phase-fraction conditioned) | 0.0414 ± 0.0046 | 0.3530 ± 0.0025 | 1.9173 ± 0.0642 | — | — |
| 3D (anchored, 25%) | 0.0293 ± 0.0033 | 0.3405 ± 0.0012 | 2.1486 ± 0.0116 | 94.88 ± 0.03% | — |
| 3D (anchored, 50%) | 0.0364 ± 0.0027 | 0.3210 ± 0.0012 | 2.1765 ± 0.0108 | 97.13 ± 0.06% | — |
| 3D (anchored, 75%) | 0.0369 ± 0.0026 | 0.3190 ± 0.0011 | 2.2004 ± 0.0077 | 97.23 ± 0.07% | — |
| 3D (anchored, 100%) | 0.0389 ± 0.0025 | 0.3224 ± 0.0011 | 2.1723 ± 0.0080 | 97.93 ± 0.08% | — |
| 3D (scale-up) | 0.0091 ± 0.0014 | 0.3400 ± 0.0029 | 2.0967 ± 0.0154 | — | 0.32 ± 0.22 pp |

### 5.1 Three-dimensional synthesis

The cutaway in Figure 2 exposes both the surface and interior of the fixed 128³ controlled-reference volume. Its three orthogonal center sections (Figure 3) show feature scales comparable to those in the training image. These figures provide qualitative context; Table 1 supplies the distributional and transport measurements.

<p align="center">
  <img src="assets/paper/02-generated-volume.png" alt="Generated binary 3D volume with one octant removed" width="470">
</p>
<p align="center"><em>Figure 2. Generated 128³ controlled-reference volume with one octant removed. Black denotes pore and gray denotes solid.</em></p>

<p align="center">
  <img src="assets/paper/03-generated-slices.png" alt="Three orthogonal center sections of a generated volume" width="760">
</p>
<p align="center"><em>Figure 3. Center sections normal to axes 0, 1, and 2 of the volume in Figure 2.</em></p>

### 5.2 Plane anchoring

In the seed-0 single-plane example, the generated center section matches 16,097 of 16,384 supplied voxels (98.25%) without post-generation replacement (Figure 4). This is measured learned-conditioning accuracy, not a sampler invariant. It is also distinct from the whole-volume agreement reported for multi-plane coverage in Table 1.

<p align="center">
  <img src="assets/paper/04-anchor-conditioning.png" alt="Supplied center section, generated center section, and anchored 3D volume" width="780">
</p>
<p align="center"><em>Figure 4. Fixed-seed single-plane conditioning. (a) Supplied axis-0 center section; (b) generated center section, matching 16,097/16,384 constrained voxels (98.25%); (c) surrounding 128³ volume.</em></p>

Across four seeds, whole-volume agreement is 94.88 ± 0.03% at 25% coverage, 97.13 ± 0.06% at 50%, 97.23 ± 0.07% at 75%, and 97.93 ± 0.08% at 100%. The seed-0 sweep in Figure 5 shows 55.10% agreement at zero planes, 94.87% at 32 planes, 97.14% at 64 planes, and 97.93% at 128 planes. Because each count recomputes its coordinates, this is not a nested sequence of constraints. FID is 22.18 at zero planes, 18.15 at 32 planes, and 20.24 at 128 planes; KID, porosity, and tortuosity are also non-monotonic across the sweep. The GT phase-0 pore-percolating fraction is 99.751% on every axis. Pore-percolation error is 0.081 pp at zero planes and is 0.368, 0.118, and 0.150 pp at 32, 64, and 128 planes, respectively; across all counts it ranges from 0.015 to 0.368 pp and is non-monotonic with coverage. In every evaluated volume, one pore component spans all three axes, so the three axis-specific fractions are equal.

<p align="center">
  <img src="assets/paper/06-anchor-sweep-metrics.png" alt="Voxel accuracy, KID, FID, porosity, tortuosity, and pore-percolation error versus supplied axis-0 planes" width="780">
</p>
<p align="center"><em>Figure 5. Fixed-seed controlled-reference diagnostic for 0–128 supplied axis-0 planes. (a) Voxel accuracy is measured over the complete 128³ volume. (b) KID and (c) FID compare all 128 generated axis-0 sections with all 128 synthetic-GT sections and are not directly comparable with Table 1; the unbiased KID estimate may be slightly negative. Dashed lines in (d,e) show GT porosity and tortuosity. (f) Pore-percolation error is the three-axis mean absolute prediction–GT difference in the fraction of phase-0 voxels belonging to 6-connected components that span opposing faces; connectivity is non-periodic and the dashed line marks the ideal value zero. The symmetric-log x-axis separates low anchor counts; no across-volume uncertainty is shown because the seed is fixed.</em></p>

### 5.3 Anchored scale-up

The shared-state sampler combines 3 × 3 × 3 fixed 128³ blocks with eight-voxel inward margins to produce a 352³ output, 20.80 times the base voxel count. In the illustrated seed-0 center plane, the complete 128 × 128 anchor retains 15,916 of 16,384 voxels (97.14%) after scale-up. The 120 × 120 full-strength conditioning interior retains 14,136 of 14,400 voxels (98.17%), and the corresponding 120³ base interior retains 1,727,293 of 1,728,000 voxels (99.96%). Across four samples, local pore continuation at tile boundaries is lower than in the interior by 0.32 ± 0.22 percentage points on average.

<p align="center">
  <img src="assets/paper/05-scale-up.png" alt="Anchor, scaled center section, and cutaway of a 352 cubed volume" width="780">
</p>
<p align="center"><em>Figure 6. Scale-up with 3 × 3 × 3 fixed 128³ blocks and eight-voxel inward margins on shared faces. (a) 128² center-plane anchor; (b) 352² center section, with the orange box marking the full 128² base footprint and the blue dashed box marking the 120² full-strength conditioning region; (c) cutaway of the 352³ output. No base voxels are pasted into the result.</em></p>

### 5.4 Overlap ablation

Table 2 isolates one boundary between two fixed 128³ blocks and reports mean ± sample standard deviation over five seeds. Inward margins of 0, 4, 8, 12, and 16 voxels give axis-0 output lengths of 256, 248, 240, 232, and 224. The exact seam-change ratios are 6.90, 1.12, 0.99, 0.90, and 0.98; transition TV values are 0.380, 0.064, 0.043, 0.041, and 0.047; and continuation deltas are 0.567, 0.058, 0.022, 0.021, and 0.032. Runtime remains 1.74–1.84 s and peak allocated memory is 0.842–0.851 GiB. The default eight-voxel margin is closest to the ideal exact-seam ratio of one while retaining low band discrepancies.

| Inward margin per shared face | Output shape | Exact seam-change ratio | Transition TV | Continuation delta | Time | Peak GPU memory |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 256 × 128 × 128 | 6.90 ± 0.52 | 0.380 ± 0.031 | 0.567 ± 0.057 | 1.844 ± 0.224 s | 0.851 ± 0.000 GiB |
| 4 | 248 × 128 × 128 | 1.12 ± 0.11 | 0.064 ± 0.016 | 0.058 ± 0.030 | 1.755 ± 0.038 s | 0.851 ± 0.000 GiB |
| 8 | 240 × 128 × 128 | 0.99 ± 0.16 | 0.043 ± 0.017 | 0.022 ± 0.003 | 1.735 ± 0.011 s | 0.849 ± 0.000 GiB |
| 12 | 232 × 128 × 128 | 0.90 ± 0.09 | 0.041 ± 0.009 | 0.021 ± 0.007 | 1.737 ± 0.017 s | 0.845 ± 0.000 GiB |
| 16 | 224 × 128 × 128 | 0.98 ± 0.14 | 0.047 ± 0.006 | 0.032 ± 0.010 | 1.738 ± 0.016 s | 0.842 ± 0.000 GiB |

<p align="center">
  <img src="assets/paper/07-overlap-ablation.png" alt="Scale-up overlap ablation showing boundary quality, time, and memory" width="780">
</p>
<p align="center"><em>Figure 7. Five-seed fixed-two-block overlap ablation; axis-0 output lengths are 256, 248, 240, 232, and 224 voxels for inward margins 0, 4, 8, 12, and 16, while both transverse axes remain 128. Error bars show sample standard deviations. (a) Exact seam phase-change rate relative to the median interior rate; the dashed line marks one. (b) Maximum transition-distribution and phase-continuation discrepancies within the seam band; lower is better. (c) Elapsed generation time and peak allocated GPU memory on an RTX 2060.</em></p>

## 6. Discussion

The controlled same-model sweep gives whole-volume accuracies of 55.10%, 94.87%, 97.14%, and 97.93% at 0, 32, 64, and 128 supplied planes. FID at the same counts is 22.18, 18.15, 18.48, and 20.24. KID, porosity, and tortuosity are shown for all nine counts in Figure 5. Coordinates are recomputed for each count rather than nested across counts.

The GT pore-percolating fraction is 99.751% on axes 0, 1, and 2. For anchor counts 0, 1, 2, 4, 8, 16, 32, 64, and 128, pore-percolation errors are 0.081, 0.056, 0.015, 0.092, 0.115, 0.204, 0.368, 0.118, and 0.150 pp. In each evaluated volume, the three axis-specific fractions are equal and one pore component spans all three axes.

The fixed-block sampler uses 27 denoiser blocks to produce a 352³ volume, 20.80 times the base voxel count. Across four seeds, its local pore-continuation drop is 0.32 ± 0.22 pp. In the two-block ablation, the default eight-voxel margin gives an exact seam-change ratio of 0.99 ± 0.16, transition TV of 0.043 ± 0.017, and continuation delta of 0.022 ± 0.003; without overlap, the corresponding values are 6.90 ± 0.52, 0.380 ± 0.031, and 0.567 ± 0.057.

## 7. Limitations

The synthetic GT is sampled from the trained model and therefore lies within its learned distribution. It tests controlled coordinate agreement, not reconstruction of unseen experimental 3D material. The real crop baseline, training data, and single-plane anchor all come from the same 2D image, with no held-out specimen. The image's material provenance, imaging and segmentation history, physical scale, and external license are unavailable.

The anchor study measures supplied-plane and whole-volume agreement but provides no exact label guarantee, comparison with a slice-conditioned baseline, or ablation of the anchor input, anchor loss, connectivity critic, normal-transition loss, or shared-state fusion. Pore percolation is aggregated over the complete volume and does not isolate topology specifically around anchor planes. Because the connectivity targets are unconditioned model triplets rather than measured 3D neighborhoods, their contribution to physical connectivity cannot be inferred from the present results. The fixed-block results reuse the existing checkpoint rather than a model retrained under the current fixed-128 training recipe; they therefore isolate inference geometry, not retraining effects.

The quantitative evaluation uses one binary image, four seeded replicates for Table 1, one oracle phase-fraction target, one fixed-seed 128³ anchor sweep, and five seeds for the two-block overlap sweep. Because block count is fixed while the inward margin changes, the ablation's axis-0 output length decreases from 256 to 224 voxels; its runtime and memory values are geometry-specific rather than pure overlap-cost estimates. Pore percolation reports the fraction of phase-0 voxels in non-periodic 6-connected components that span opposite faces. Conductivity and permeability are not calculated by this metric. KID and FID use Inception-v3 features, and runtime and memory are measured on the reported hardware.

## 8. Conclusion

This work combines plane-anchor conditioning with fixed-block shared-state diffusion in a 2D-supervised categorical 3D generator. In the same-model controlled experiment, 25% and 100% axis-0 coverage yield 94.88 ± 0.03% and 97.93 ± 0.08% whole-volume agreement. The 128-plane pore-percolation error is 0.150 pp. The 3 × 3 × 3 sampler produces a 352³ output from fixed 128³ blocks; its local pore-continuation drop is 0.32 ± 0.22 pp across four seeds. With the default eight-voxel inward margin, the exact seam-change ratio is 0.99 ± 0.16, compared with 6.90 ± 0.52 without overlap.

## References

[1] R. Bostanabad, “Reconstruction of 3D microstructures from 2D images via transfer learning,” *Computer-Aided Design*, vol. 128, art. 102906, 2020.

[2] J. Feng, Q. Teng, B. Li, X. He, H. Chen, and Y. Li, “An end-to-end three-dimensional reconstruction framework of porous media from a single two-dimensional image based on deep learning,” *Computer Methods in Applied Mechanics and Engineering*, vol. 368, art. 113043, 2020.

[3] S. Kench and S. J. Cooper, “Generating three-dimensional structures from a two-dimensional slice with generative adversarial network-based dimensionality expansion,” *Nature Machine Intelligence*, vol. 3, pp. 299–305, 2021.

[4] J. Phan et al., “Generating 3D images of material microstructures from a single 2D image: a denoising diffusion approach,” *Scientific Reports*, vol. 14, art. 6498, 2024.

[5] K.-H. Lee and G. J. Yun, “Multi-plane denoising diffusion-based dimensionality expansion for 2D-to-3D reconstruction of microstructures with harmonized sampling,” *npj Computational Materials*, vol. 10, art. 99, 2024.

[6] O. Bar-Tal, L. Yariv, Y. Lipman, and T. Dekel, “MultiDiffusion: Fusing diffusion paths for controlled image generation,” in *Proceedings of the 40th International Conference on Machine Learning*, vol. 202, pp. 1737–1752, 2023.

[7] C. Düreth, P. Seibert, D. Rücker, S. Handford, M. Kästner, and M. Gude, “Conditional diffusion-based microstructure reconstruction,” *Materials Today Communications*, vol. 35, art. 105608, 2023.

[8] A. Sadeghkhani, B. Bennett, and A. Rabbani, “Property-constrained 3D porous media reconstruction from 2D images via conditional generative adversarial networks,” in *87th EAGE Annual Conference & Exhibition*, vol. 2026, pp. 1–5, 2026.

[9] Z. Ma et al., “A sliced-Wasserstein and neural network framework for statistically controllable 3D microstructure reconstruction,” *Computer-Aided Design*, vol. 198, art. 104100, 2026.

[10] S. Fan, Y. Li, X. Wang, and D. Du, “Dual-domain latent diffusion framework with multi-modal conditioning for controllable porous media reconstruction,” *Computational Materials Science*, vol. 272, art. 114876, 2026.

[11] Y. Shi et al., “GeoTopoDiff: Learning geometry–topology graph priors through boundary-constrained mixed diffusion for sparse-slice 3D porous reconstruction,” *arXiv preprint* arXiv:2605.03764, 2026.

[12] G. Zhang et al., “Towards coherent image inpainting using denoising diffusion implicit models,” in *Proceedings of the 40th International Conference on Machine Learning*, vol. 202, pp. 41164–41193, 2023.

[13] A. Lugmayr et al., “RePaint: Inpainting using denoising diffusion probabilistic models,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 11461–11471, 2022.

[14] Z. Ding, M. Zhang, J. Wu, and Z. Tu, “Patched denoising diffusion models for high-resolution image synthesis,” in *International Conference on Learning Representations*, 2024.

[15] N. Hoffman, C. Diniz, D. Liu, T. M. Rodgers, A. Tran, and M. D. Fuge, “GrainPaint: A multi-scale diffusion-based generative model for microstructure reconstruction of large-scale objects,” *Acta Materialia*, vol. 288, art. 120784, 2025.

[16] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in *Advances in Neural Information Processing Systems*, vol. 33, pp. 6840–6851, 2020.

[17] Z. Xiao, K. Kreis, and A. Vahdat, “Tackling the generative learning trilemma with denoising diffusion GANs,” in *International Conference on Learning Representations*, 2022.

[18] M. Bińkowski, D. J. Sutherland, M. Arbel, and A. Gretton, “Demystifying MMD GANs,” in *International Conference on Learning Representations*, 2018.

[19] S. J. Cooper, A. Bertei, P. R. Shearing, J. A. Kilner, and N. P. Brandon, “TauFactor: An open-source application for calculating tortuosity factors from tomographic data,” *SoftwareX*, vol. 5, pp. 203–210, 2016.

[20] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “GANs trained by a two time-scale update rule converge to a local Nash equilibrium,” in *Advances in Neural Information Processing Systems*, vol. 30, 2017.
