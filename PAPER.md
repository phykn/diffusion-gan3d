<div align="center">
  <h1>Anchor-Conditioned Diffusion for Scalable 3D Microstructure Synthesis</h1>
  <p><strong>Kwangnam Yu</strong></p>
</div>

## Abstract

Material simulations need sufficiently large 3D microstructures to represent connected pores and transport paths, but acquiring such volumes experimentally is expensive. Two-dimensional microscopy is much easier to obtain, so we train a 3D generator using independent 2D sections viewed from three perpendicular directions. This recovers the overall appearance of the material, but a measured section must also remain at its known position. Simply pasting it into a generated volume can break structures on either side of the section. Our method instead provides the measured section and its location to the generator during every denoising step. It directly penalizes incorrect phase labels on the section and examines crossing sections to encourage a continuous 3D neighborhood around it. We also generate volumes larger than the training size. Rather than completing small blocks independently and stitching them together, overlapping blocks are denoised as parts of one shared volume and their predictions are blended at every step. In a deterministic coverage sweep, whole-volume voxel accuracy increased from 54.93% without measured sections to 98.26% when all axis-0 sections were supplied. A 3 × 3 × 3 expansion produced a 192³ volume while preserving 100.00% of the protected central region. The method therefore turns sparse 2D observations into large 3D microstructures that retain measured information and connected morphology.

## 1. Introduction

Digital 3D microstructures provide the geometric domains used to study permeability, diffusion, and other connectivity-dependent material behavior. These predictions require volumes that are both representative in size and consistent with available measurements. Experimental 3D imaging, however, is costly and often limited in either resolution or field of view, whereas high-quality 2D sections are much easier to collect.

Methods such as SliceGAN [1] address this data gap by training a 3D generator against 2D sections. More recent diffusion-based dimensional-expansion methods likewise reconstruct 3D structure from one or more 2D views [2–4]. Their main objective is statistical reconstruction: generated sections should resemble the training images. This is sufficient for producing representative samples, but not for reconstructing around a particular measured section at a known location. A statistically plausible volume may still contradict that measurement, while replacing a generated plane afterward can sever pores and create an artificial transport barrier.

We treat a measured section as a spatial constraint rather than only as a training example. The section, its orientation, and its position are supplied to the 3D generator throughout denoising. A direct phase-label loss encourages agreement on the constrained voxels, and 2D critics inspect orthogonal sections that cross the constraint so that the surrounding morphology connects to it. Because these signals are constructed from the same independent 2D datasets used for ordinary training, the method does not require paired 2D–3D examples or ground-truth training volumes.

A second challenge is output size. A model trained on small patches cannot directly cover the larger domains needed for simulation, while independently generated blocks generally disagree at their boundaries. Inspired by shared-path tiled diffusion methods such as MultiDiffusion [6], we maintain one noisy 3D state for the complete output. Overlapping tiles predict the same regions, their predictions are blended, and the shared volume is updated only after this fusion. The process can grow around an anchored base while retaining its inner region and allowing its boundary to adapt to the newly generated surroundings.

### Contributions

The individual ideas of masked diffusion conditioning, boundary-aware adversarial learning, and overlapping tiled denoising are not claimed as novel in isolation. The contribution of this work is their specialization and integration for categorical 3D microstructure synthesis from unpaired 2D data:

1. **Plane-anchored reconstruction without paired 3D supervision.** A 2D-supervised diffusion–GAN is extended to reconstruct a categorical 3D neighborhood around one or more measured internal sections placed at explicit axes and coordinates. The image-mask condition, anchor-voxel loss, and seam-focused orthogonal critics are trained using only unaligned 2D section datasets.
2. **Anchor-preserving scalable synthesis.** The anchored result can be embedded into a larger jointly denoised volume. A retained inner core preserves the supplied structure, an adaptive shell connects it to newly generated material, and overlapping 3D tile predictions are fused at every reverse transition rather than stitching independently completed blocks.
3. **Controlled validation of conditioning and scale-up.** Anchor coverage is varied from zero to complete axis coverage and evaluated against the same reference volume, separating whole-volume voxel fidelity from section-distribution similarity, porosity, and transport behavior. The scale-up experiment additionally measures continuity across block boundaries.

## 2. Related Work

### 2.1 3D microstructure reconstruction from 2D data

SliceGAN showed that a 3D generator can be trained without 3D examples by asking 2D discriminators to judge generated cross-sections [1]. This dimensional-expansion strategy makes 3D synthesis possible from a single representative micrograph and supports large outputs through a fully convolutional generator. Diffusion models have since been applied to microstructure reconstruction because their iterative sampling provides a stable alternative to direct adversarial generation [2,3]. Multi-plane diffusion further uses sections from different orientations to improve consistency between the three spatial directions [4]. Other conditional approaches control global descriptors such as phase fraction or spatial statistics [8,9]. These methods aim primarily at statistical equivalence: the generated sections or descriptors should follow the reference distribution. They do not directly address the separate requirement that a particular measured section must appear at a specified coordinate inside the generated volume.

### 2.2 Conditioning on measured regions

Image inpainting provides a related form of spatial conditioning. RePaint repeatedly restores known pixels during reverse diffusion so that an unconditional model can fill an unknown region [10], while coherent diffusion inpainting methods explicitly reduce disagreement at the boundary between known and generated content [7]. Such methods establish that a mask can preserve observations and that boundary treatment is essential. An internal microstructure section presents a different geometric constraint: it is a thin categorical plane surrounded by unknown material on both sides, and its quality depends on 3D connections that are visible only in crossing sections. The present work adapts masked conditioning to this setting by learning the plane condition together with phase-label supervision and adversarial evaluation of orthogonal seams, using only unpaired 2D micrographs for training.

### 2.3 Scalable and tiled diffusion generation

Diffusion models are commonly trained at a fixed spatial size because storing and denoising an entire high-resolution state is expensive. MultiDiffusion addresses this limitation in 2D by applying a pretrained model to overlapping crops and combining their denoising paths into one output [6]. Patch-DM likewise uses overlapping patch information to suppress boundaries in high-resolution image synthesis [12]. For microstructures, GrainPaint expands generation domains through diffusion inpainting [11]. These studies show that overlap must be resolved during generation rather than after independently completed patches are stitched together. Our scale-up procedure follows this principle in a categorical 3D setting: all tiles read from a shared noisy volume, overlapping clean predictions are cosine-fused before each posterior update, and an optional anchored base is connected through a protected core and a gradual transition shell.

## 3. Method

### 3.1 Problem formulation

Let $\mathcal{D}_a$ be a dataset of categorical 2D sections normal to axis $a \in \{0,1,2\}$. Each pixel is a phase label in $\{0,\ldots,K-1\}$, where $K$ is the number of material phases. The three datasets are independent: images from different axes do not need to depict the same specimen or spatial location. Our goal is to learn a generator for a categorical volume $X \in \{0,\ldots,K-1\}^{D \times H \times W}$ whose sections match the corresponding 2D distributions. At inference, the generator may additionally receive measured full or partial planes with specified orientations and coordinates. These planes constrain selected voxels, while the remaining volume is generated stochastically.

### 3.2 2D-supervised 3D generation

The generator is a 3D denoising network $G_\theta(x_t,t,z,c)$, where $x_t$ is the current noisy volume, $t$ is the diffusion state, $z$ is a newly sampled latent vector, and $c$ contains any active conditions. The network predicts $K$ logits per voxel. A softmax converts these logits into phase probabilities, which are mapped to $[-1,1]$ to obtain a clean-volume estimate $\hat{x}_0$. Each reverse step samples the next state from the diffusion posterior $q(x_{t-1}\mid x_t,\hat{x}_0)$. Following the short denoising diffusion-GAN formulation [5], adversarial training makes it possible to use only a small number of reverse transitions.

Because no real 3D volumes are available, supervision is applied through three 2D critics $C_a$, one for each axis. A critic receives consecutive noisy section states $(x_{t-1}^{(a)},x_t^{(a)})$. Real pairs are produced by forward-noising images from $\mathcal{D}_a$, while generated pairs are sliced from the 3D reverse process. Each critic has a global head for whole-section structure and a local head for phase boundaries. The base generator objective is

$$
\mathcal{L}_{\mathrm{adv}}
=\sum_{a=0}^{2}\left(\mathcal{L}_{\mathrm{global}}^{(a)}
+\lambda_{\mathrm{local}}\mathcal{L}_{\mathrm{local}}^{(a)}\right).
$$

This arrangement constrains 3D generation from all three directions while requiring only unaligned 2D examples.

### 3.3 Plane-anchor conditioning

An anchor is defined by a categorical image, its normal axis, its index on that axis, and an optional in-plane offset. Multiple anchors are assembled into a target tensor $Y$ and a binary mask $M$ over the generated volume. $M(v)=1$ indicates that voxel $v$ is constrained; elsewhere the target is ignored. Full planes and smaller rectangular patches use the same representation.

The masked one-hot anchor and its mask are passed through a zero-initialized 3D convolution and added to the first generator feature map:

$$
h_A=\operatorname{Conv}_{3D}\!\left(\left[\operatorname{onehot}(Y)\odot M,\;M\right]\right).
$$

The zero initialization preserves the unconditioned network at the start of training. The same anchor features are supplied at every reverse step, allowing the model to organize the surrounding structure around the measured phases rather than inserting them after generation. Agreement on constrained voxels is learned with

$$
\mathcal{L}_{\mathrm{anchor}}
=-\frac{1}{|M|}\sum_{v:M(v)=1}
\log p_\theta\!\left(Y(v)\mid x_t,t,z,M\right).
$$

Training randomly selects the anchor axis, plane positions, and source crops. Anchors in one training sample share an axis so that unrelated images from independent datasets do not impose contradictory intersection labels. Anchor dropout retains the ability to generate without a constraint.

### 3.4 Seam-aware training

The voxel loss alone does not ensure that structures connect across an anchored plane. We therefore sample additional generated sections whose normals are orthogonal to the anchor and whose crop centers intersect its mask. The corresponding critics evaluate these crossing sections at the clean transition, producing a seam adversarial loss $\mathcal{L}_{\mathrm{seam}}$. This loss exposes discontinuities on both sides of the anchor even though the training data contain no paired 3D neighborhood.

With optional phase-fraction conditioning, the complete generator objective is

$$
\mathcal{L}_{G}=\mathcal{L}_{\mathrm{adv}}
+\lambda_{\mathrm{anchor}}\mathcal{L}_{\mathrm{anchor}}
+\lambda_{\mathrm{seam}}\mathcal{L}_{\mathrm{seam}}
+\lambda_{\mathrm{vf}}\mathcal{L}_{\mathrm{vf}}.
$$

### 3.5 Phase-fraction conditioning

An optional vector $v\in[0,1]^K$, normalized so that $\sum_k v_k=1$, controls the desired phase composition. Its embedding is added to the time and latent embeddings used by the denoiser. If $\hat{p}$ denotes the spatial mean of the predicted phase probabilities, the conditioning loss is

$$
\mathcal{L}_{\mathrm{vf}}=\lVert\hat{p}-v\rVert_1.
$$

Dropping this condition for a fraction of training samples allows conditioned and unconditioned generation with the same model. Phase-fraction and plane-anchor conditions can be active simultaneously.

### 3.6 Joint tiled scale-up

Let $P$ be the training patch size and $o$ the overlap on each side. Scale-up covers the requested output with tiles of input size $(P+2o)^3$ whose core regions are spaced $P$ voxels apart. Unlike independent block generation, every tile reads from one shared noisy volume $x_t$. At reverse step $t$, tile $k$ predicts a clean volume $\hat{x}_{0,k}$. A separable cosine-taper window $w_k$ blends all predictions that cover voxel $v$:

$$
\bar{x}_0(v)=
\frac{\sum_k w_k(v)\hat{x}_{0,k}(v)}
{\sum_k w_k(v)}.
$$

Only after this fusion is the shared state updated through $q(x_{t-1}\mid x_t,\bar{x}_0)$. Consequently, neighboring tiles repeatedly reconcile their overlap throughout denoising instead of meeting for the first time in the final output.

An optional base volume is placed at the center of the global state. Its inner core is retained, while a cosine-weighted shell gradually reduces the constraint toward the boundary so the generated surroundings can connect to it. The complete base is preserved when $o=0$. For outputs that exceed GPU memory, the global states remain in CPU memory and only the active tile is transferred to the generator; this changes storage location but not the update rule.

## 4. Experimental Setup

### 4.1 Data and preprocessing

The training data consist of a binary phase map of size 226 × 690 pixels. Phase 0 denotes pore space and phase 1 denotes solid material. Because only one section orientation is available, the same image pool is used for all three axes, which assumes isotropic section statistics. During training, 128 × 128 regions are sampled at random, resized to 64 × 64 with nearest-neighbor interpolation, and augmented by rotations and reflections. Figure 1 shows the source image and example crop regions.

<p align="center">
  <img src="assets/paper/01-training-data.png" alt="Categorical 2D training data with 128 by 128 crop regions" width="680">
</p>
<p align="center"><em>Figure 1. Binary training micrograph. Orange boxes indicate example 128 × 128 training crops.</em></p>

For the final distribution-level evaluation, the real 2D set is defined as 64 independently sampled 128 × 128 crops from the source image. The crops are resized to the model resolution before comparison with generated sections. The 64³ volume in `scripts/gt.tiff` is used only as a controlled reference for the anchor-coverage experiment; it is not treated as experimentally measured 3D ground truth.

### 4.2 Training configuration

The 3D denoiser uses base width 16, channel multipliers (1, 2, 4, 4), a 128-dimensional embedding, and a 64-dimensional latent vector. Training alternates between 64³ and 96³ generated volumes and samples 16 sections per axis. The diffusion process uses 10 transitions with beta limits of 0.1 and 20. The generator and critics are optimized for 30,000 steps with learning rates of 1.6 × 10⁻⁴ and 1.0 × 10⁻⁴, respectively, Adam coefficients (0.5, 0.9), mixed-precision arithmetic, and an exponential moving average of 0.999. The resolved settings are summarized in Table 1.

| Setting | Value |
|---|---:|
| Number of phases | 2 |
| Source crop / model patch | 128² / 64² |
| 2D batch size / 3D volume batch size | 8 / 1 |
| Training volume sizes | 64³, 96³ |
| Diffusion transitions | 10 |
| Training steps | 30,000 |
| Maximum anchor planes per training sample | 4 |
| Anchor dropout / loss weight | 0.2 / 1.0 |
| Seam loss weight | 0.25 |
| Phase-fraction dropout / loss weight | 0.2 / 1.0 |

### 4.3 Anchor and scale-up protocols

Two anchor tests are used. First, the central 128 × 128 crop from the training image is resized to 64 × 64 and placed at axis 0, index 32 of a 64³ volume. Second, complete axis-0 planes are drawn from the controlled GT volume and supplied at counts of 0, 1, 2, 4, 8, 16, 32, and 64. The selected planes are distributed through the volume, corresponding to coverages from 0% to 100%. Every condition uses the same random seed so that changes can be attributed to the anchors rather than initial noise.

The scale-up test embeds the anchored 64³ base at the center of a 192³ output. The output is generated using a 3 × 3 × 3 block arrangement with an overlap of 16 voxels. An eight-voxel transition shell is allowed to adapt, leaving a protected 48³ core.

### 4.4 Evaluation metrics

The evaluation separates section appearance, phase proportion, 3D transport, anchor fidelity, and scale-up boundaries:

- **Kernel Inception Distance (KID):** compares real and generated section distributions; lower is better. Generated axis-0 sections are compared with reference sections using 2,048-dimensional Inception features, 100 subsets, and a subset size of 50.
- **Porosity:** the fraction of all pixels or voxels assigned to phase 0. Two-dimensional porosity is an area fraction and three-dimensional porosity is a volume fraction; agreement with the reference is desired.
- **Tortuosity:** the ratio between the effective pore-path length and the straight transport distance. It is computed along axis 0 by a steady-state diffusion solve through phase-0 pore space; agreement with the reference is desired.
- **Voxel accuracy:** the fraction of all generated voxels whose phase equals the voxel at the same coordinate in GT. This whole-volume score measures how spatial reconstruction improves as anchor coverage increases; higher is better.
- **Seam connectivity drop:** the connectivity in ordinary interior regions minus the connectivity across scale-up boundaries. A value near zero indicates that tiling introduces little connectivity loss.

Table 2 defines which metrics apply to each final comparison. KID replaces FID because the available section set is small. The final evaluation should report four independently generated 3D samples per condition as mean ± standard deviation. The currently available CSV contains one deterministic volume per anchor count; therefore, its KID deviation is the deviation across KID subsets, not variation across generated volumes.

| Evaluation data | KID | Porosity | Tortuosity | Voxel accuracy | Seam connectivity drop |
|---|:---:|:---:|:---:|:---:|:---:|
| GT reference volume | — | ✓ | ✓ | — | — |
| Real 2D crops | ✓ | ✓ | — | — | — |
| 3D | ✓ | ✓ | ✓ | — | — |
| 3D (phase-fraction conditioned) | ✓ | ✓ | ✓ | — | — |
| 3D (anchored, 25%) | ✓ | ✓ | ✓ | ✓ | — |
| 3D (anchored, 50%) | ✓ | ✓ | ✓ | ✓ | — |
| 3D (anchored, 75%) | ✓ | ✓ | ✓ | ✓ | — |
| 3D (anchored, 100%) | ✓ | ✓ | ✓ | ✓ | — |
| 3D (scale-up) | ✓ | ✓ | ✓ | — | ✓ |

## 5. Results

### 5.1 Three-dimensional generation from 2D data

The model produces a binary 3D volume after training only on 2D sections. Figure 2 removes one octant to show both the exterior and the internal pore morphology. The three orthogonal center sections in Figure 3 contain feature sizes and phase patterns similar to those in the training image. These figures provide qualitative evidence of plausible 3D synthesis; the distributional and physical metrics below provide the quantitative checks.

<p align="center">
  <img src="assets/paper/02-generated-volume.png" alt="Generated 3D categorical volume with one octant removed" width="420">
</p>
<p align="center"><em>Figure 2. Generated 64³ volume with one octant removed to expose the internal structure.</em></p>

<p align="center">
  <img src="assets/paper/03-generated-slices.png" alt="Orthogonal sections of the generated 3D volume" width="720">
</p>
<p align="center"><em>Figure 3. Orthogonal center sections of the generated volume.</em></p>

### 5.2 Single-plane conditioning

In the fixed-seed center-plane test, all 4,096 constrained voxels match the supplied section, giving 100.00% anchor-plane accuracy without overwriting the generated output after sampling. The cutaway in Figure 4 shows that the model generates material on both sides of the measured plane rather than inserting the plane into an already completed volume. This exact match is a result for this reproducible example, not a guarantee for every seed and anchor geometry.

<p align="center">
  <img src="assets/paper/04-anchor-conditioning.png" alt="Input center anchor, matching generated plane, and anchored 3D volume" width="820">
</p>
<p align="center"><em>Figure 4. Supplied center section, corresponding generated section, and the surrounding anchored 3D volume.</em></p>

### 5.3 Effect of anchor coverage

Table 3 reports the values stored in `temp/anchor_sweep_metrics.csv`. Whole-volume voxel accuracy increases consistently from 54.93% with no anchors to 98.26% with all 64 axis-0 planes supplied. The strongest increases occur at low and intermediate coverage, while the final gain from 50% to 100% is smaller because most voxels already agree with GT.

| Anchor planes | Coverage | Voxel accuracy | KID | Porosity | Tortuosity |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.00% | 54.93% | 0.0104 ± 0.0014 | 0.3535 | 2.2572 |
| 1 | 1.56% | 58.41% | 0.0065 ± 0.0012 | 0.3487 | 2.1764 |
| 2 | 3.12% | 63.25% | 0.0037 ± 0.0008 | 0.3532 | 2.2478 |
| 4 | 6.25% | 70.95% | 0.0032 ± 0.0011 | 0.3616 | 2.1004 |
| 8 | 12.50% | 80.97% | 0.0242 ± 0.0040 | 0.3499 | 2.6027 |
| 16 | 25.00% | 89.16% | 0.0130 ± 0.0027 | 0.3421 | 2.4861 |
| 32 | 50.00% | 95.64% | 0.0080 ± 0.0022 | 0.3296 | 2.2954 |
| 64 | 100.00% | 98.26% | 0.0101 ± 0.0018 | 0.3402 | 2.2339 |

GT has a porosity of 0.352 and an axis-0 tortuosity of 2.224. Across the sweep, generated porosity remains between 0.330 and 0.362. Tortuosity ranges from 2.100 to 2.603, with the fully anchored volume close to the GT value. KID ranges from 0.0032 to 0.0242 and is not monotonic with anchor count. This is expected because KID compares an unordered distribution of sections, whereas voxel accuracy tests phase identity at corresponding 3D coordinates. Increasing spatial agreement with GT therefore need not reduce KID at every intermediate coverage.

<p align="center">
  <img src="assets/paper/06-anchor-sweep-metrics.png" alt="Voxel accuracy, KID, porosity, and axis-0 tortuosity across anchor-plane coverage" width="780">
</p>
<p align="center"><em>Figure 5. Quantitative effect of the number of supplied axis-0 anchor planes. Horizontal GT lines are shown for porosity and tortuosity.</em></p>

### 5.4 Anchored scale-up

The 3 × 3 × 3 scale-up produces a 192³ volume around the anchored 64³ base. The complete 64 × 64 center anchor area retains 3,809 of 4,096 voxels, or 92.99% agreement, because its transition shell is deliberately allowed to change. Within the protected 48 × 48 center region, all 2,304 voxels are preserved, corresponding to 100.00% accuracy. Figure 6 shows the embedded region within the full center section and the resulting cutaway volume. Seam connectivity drop has not yet been computed, so the figure demonstrates scale and anchor retention but does not by itself establish quantitative seam equivalence.

<p align="center">
  <img src="assets/paper/05-scale-up.png" alt="Anchored 3 by 3 by 3 scale-up with a 192 by 192 center section and 192 cubed volume" width="820">
</p>
<p align="center"><em>Figure 6. Anchored scale-up to 192³ using 27 overlapping blocks.</em></p>

## 6. Discussion

The coverage experiment separates two properties that are often combined under the term reconstruction quality. Voxel accuracy measures whether the model recovers a particular reference volume at the correct coordinates, while KID measures whether its sections belong to a similar image distribution regardless of position. The monotonic accuracy increase confirms that additional planes provide useful spatial information. The non-monotonic KID values do not contradict this result: a volume can become more accurate voxel by voxel while its finite set of sections changes only slightly, or temporarily moves within the natural variation of the reference distribution.

Porosity is comparatively stable because it depends only on the total number of pore voxels. Tortuosity is more sensitive to how those voxels connect across the volume, which explains its larger variation at sparse and intermediate coverage. The approach therefore does more than copy the global phase fraction: dense anchors progressively recover spatial arrangement and bring the axis-0 transport response closer to GT. At the same time, full coverage reaches 98.26% rather than exactly 100%, reflecting the use of learned soft conditioning instead of post-generation replacement.

The scale-up result illustrates a related trade-off. Freezing the complete base would preserve every voxel but could create a sharp interface with the new surroundings. The adaptive shell sacrifices some agreement near the base boundary, while the protected core remains essentially unchanged. Joint denoising then allows overlapping blocks to negotiate shared regions throughout generation. The present result supports anchor retention at larger scale; a seam-connectivity measurement is still required to determine how well the new regions connect across every block boundary.

More broadly, the method should be interpreted as conditional synthesis rather than unique 3D recovery. Sparse sections reduce the set of admissible volumes, but regions between them remain stochastic. This is useful when a measured section must be honored while multiple plausible 3D continuations are needed for uncertainty analysis.

## 7. Limitations

The present study has several limitations. First, unpaired 2D statistics cannot uniquely determine a 3D structure. The controlled GT volume is itself a generated reference used to verify anchor behavior, not an experimental tomographic volume. The current quantitative CSV also contains only one generated volume per anchor count and one random seed. Consequently, the KID deviations in Table 3 describe subset sampling and must not be interpreted as variation across independent generated volumes.

Second, the experiments use one binary, isotropic training image and evaluate anchors only along axis 0. Tortuosity is likewise measured only along axis 0 through phase-0 pore space. Multi-axis anchors, intersecting measurements, anisotropic datasets, additional phase counts, and transport in the other directions remain to be tested. Inception features used by KID were learned from natural images and may not capture every morphology relevant to porous media.

Finally, the scale-up demonstration covers one 192³ output with one overlap setting. Seam connectivity drop, repeated scale-up samples, and ablations over overlap width, seam loss, and shell thickness are not yet available. These tests are needed before claiming that block boundaries preserve transport connectivity as reliably as ordinary interior regions.

## 8. Conclusion

This work presents a 2D-supervised framework for generating categorical 3D microstructures while preserving measured internal sections and expanding beyond the training volume size. Plane anchors are supplied throughout denoising, direct phase supervision enforces their content, orthogonal critics encourage compatible surroundings, and overlapping blocks are fused through a shared reverse process during scale-up.

In the controlled anchor sweep, whole-volume voxel accuracy increases from 54.93% without anchors to 98.26% at complete axis-0 coverage. The 192³ scale-up retains 100.00% of the protected center region while allowing its outer shell to adapt. These results establish the feasibility of anchor-conditioned and scalable generation, but experimental 3D validation, repeated-sample statistics, multi-axis tests, and seam-connectivity measurements are still required to establish generality and physical reliability.

## References

[1] S. Kench and S. J. Cooper, “Generating three-dimensional structures from a two-dimensional slice with generative adversarial network-based dimensionality expansion,” *Nature Machine Intelligence*, vol. 3, pp. 299–305, 2021.

[2] C. Düreth et al., “Conditional diffusion-based microstructure reconstruction,” *arXiv preprint arXiv:2211.13497*, 2022.

[3] J. Phan et al., “Generating 3D images of material microstructures from a single 2D image: a denoising diffusion approach,” *Scientific Reports*, vol. 14, art. 6498, 2024.

[4] K.-H. Lee and G. J. Yun, “Multi-plane denoising diffusion-based dimensionality expansion for 2D-to-3D reconstruction of microstructures with harmonized sampling,” *npj Computational Materials*, vol. 10, art. 99, 2024.

[5] Z. Xiao, K. Kreis, and A. Vahdat, “Tackling the generative learning trilemma with denoising diffusion GANs,” in *International Conference on Learning Representations*, 2022.

[6] O. Bar-Tal et al., “MultiDiffusion: Fusing diffusion paths for controlled image generation,” in *Proceedings of the 40th International Conference on Machine Learning*, vol. 202, pp. 1737–1752, 2023.

[7] G. Zhang et al., “Towards coherent image inpainting using denoising diffusion implicit models,” in *Proceedings of the 40th International Conference on Machine Learning*, vol. 202, pp. 41164–41193, 2023.

[8] A. Sadeghkhani, B. Bennett, and A. Rabbani, “Property-constrained 3D porous media reconstruction from 2D images via conditional generative adversarial networks,” *arXiv preprint arXiv:2607.02693*, 2026.

[9] Z. Ma et al., “A sliced-Wasserstein and neural network framework for statistically controllable 3D microstructure reconstruction,” *Computer-Aided Design*, vol. 198, art. 104100, 2026.

[10] A. Lugmayr et al., “RePaint: Inpainting using denoising diffusion probabilistic models,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 11461–11471, 2022.

[11] N. Hoffman et al., “GrainPaint: A multi-scale diffusion-based generative model for microstructure reconstruction of large-scale objects,” *arXiv preprint arXiv:2503.04776*, 2025.

[12] Z. Ding, M. Zhang, J. Wu, and Z. Tu, “Patched denoising diffusion models for high-resolution image synthesis,” in *International Conference on Learning Representations*, 2024.
