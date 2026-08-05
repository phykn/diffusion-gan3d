<div align="center">
  <h1>Anchor-Conditioned Diffusion for Scalable 3D Microstructure Synthesis</h1>
</div>

> **Archived experiment note (2026-08-05):** The manuscript below documents the earlier 64³ proof-of-concept and its reported results. The current implementation no longer restores a hard inner core during scale-up, and its anchor curriculum now learns three-slice connectivity from unconditional volumes before introducing density-based, potentially mixed-axis teacher anchors. The method descriptions, figures, and numerical results below have not been regenerated for this implementation and must not be treated as its evaluation.

## Abstract

Three-dimensional microstructures must be large enough to represent pore connectivity and transport, yet volumetric imaging is often more costly than acquiring 2D sections. Existing 2D-supervised generators can reproduce section statistics but cannot ensure that a measured section appears at a prescribed location; independently generated blocks can also introduce discontinuities when the volume is enlarged. We address both problems with anchor-conditioned diffusion. Masked categorical sections and their coordinates are supplied at every reverse step, constrained voxels receive direct phase supervision, and orthogonal critics assess the surrounding morphology. For scale-up, overlapping blocks predict one shared noisy volume and their estimates are fused before each update. A protected base core is retained exactly, while a soft outer shell adapts to the generated surroundings. In a controlled fixed-seed sweep, whole-volume voxel accuracy rose from 54.93% without anchors to 98.26% with complete axis-0 coverage. A 3 × 3 × 3 expansion generated a 192³ volume—27 times the base voxel count—with a local pore-continuation drop of −0.68 ± 0.13 percentage points at tile boundaries. The results demonstrate coordinate-aware, scalable 3D synthesis from 2D observations while identifying the need for validation on experimental 3D data.

## 1. Introduction

Transport and mechanical response depend on three-dimensional morphology, not only on the appearance of individual sections. Digital material studies therefore require volumes that are large enough to contain representative connected paths. Tomography can provide such data, but its cost, resolution, and field of view often make 2D microscopy the more accessible source of microstructural information.

Optimization, adversarial, and diffusion methods can infer statistically plausible 3D structures from one or more 2D images [1–5]. Their usual objective is distributional: generated sections should resemble the observed 2D population. This does not guarantee that a particular measured section is retained at a known coordinate. Inserting that section after generation satisfies the local labels but can break structures on either side of the plane.

Volume size creates a second difficulty. A model trained on small patches cannot directly represent a much larger field of view, whereas generating blocks independently and stitching them afterward leaves no mechanism for reconciling their boundaries. Shared-state diffusion offers a way to couple overlapping predictions during sampling [6].

We combine these ideas in a categorical 3D generator trained without paired 2D–3D data. The method (i) integrates full or partial plane anchors throughout denoising, with direct label supervision and cross-plane adversarial evaluation; (ii) enlarges volumes by jointly denoising overlapping 3D blocks around a protected core; and (iii) evaluates spatial recovery, section distribution, phase fraction, diffusive tortuosity, and local boundary continuity separately. This separation is important because a volume can match 2D statistics without recovering the correct coordinates, or preserve coordinates while changing global morphology.

## 2. Related Work

### 2.1 Reconstruction from 2D observations

Feature-matching and conditional neural methods reconstruct 3D microstructures from a 2D exemplar [1,2]. SliceGAN removes the need for paired 3D supervision by applying 2D discriminators to sections of a generated volume [3]. Diffusion has also been used for 2D microstructure synthesis [7] and, more recently, for 2D-to-3D dimensional expansion [4,5]. Property-conditioned approaches control quantities such as phase fraction or spatial statistics [8,9]. These methods constrain distributions or global properties, but they do not preserve a specified internal section at a known location.

### 2.2 Conditioning on known regions

Diffusion inpainting preserves observed pixels while generating their surroundings [10,11]. A measured internal section poses a stricter 3D problem: unknown material lies on both sides, and structures must remain compatible across the plane. We adapt masked conditioning to categorical internal sections and evaluate orthogonal slices that cross the constrained region.

### 2.3 Generation beyond the training size

MultiDiffusion and Patch-DM couple overlapping denoising paths to synthesize outputs larger than the training domain [6,13]. GrainPaint applies diffusion inpainting to large microstructures [12]. Our scale-up procedure extends shared-state fusion to 3D categorical volumes and adds a hard inner core with a soft transition shell.

## 3. Method

### 3.1 Task definition

Let $\mathcal{D}_a$ be a set of categorical 2D sections normal to axis $a \in \{0,1,2\}$, with phase labels in $\{0,\ldots,K-1\}$. The axis-specific sets need not be spatially aligned. We learn a volume

$$
X \in \{0,\ldots,K-1\}^{D \times H \times W}
$$

whose sections match the corresponding 2D distributions. At inference, any subset of voxels on one or more planes may be supplied as spatial constraints; the remaining volume is sampled stochastically.

### 3.2 2D-supervised 3D diffusion

The generator is a 3D denoising network $G_\theta(x_t,t,z,c)$ that predicts a clean categorical volume $\hat{x}_0$ from noisy state $x_t$, diffusion step $t$, latent vector $z$, and optional conditions $c$. It follows the forward–reverse diffusion formulation [14] and the short adversarial reverse process of denoising diffusion GANs [15]. The denoiser uses channel widths 16, 32, 64, and 64, a 128-dimensional conditioning embedding, and a 64-dimensional latent vector.

Because paired 3D targets are unavailable, three 2D critics $C_a$ supervise orthogonal sections. Each critic distinguishes forward-noised real section pairs from pairs sliced from the generated reverse process. A global head evaluates the whole section, and a patch head evaluates fine-scale structure. The generator objective is

$$
\mathcal{L}_{\mathrm{adv}}
=\sum_{a=0}^{2}\left(
\mathcal{L}_{\mathrm{global}}^{(a)}
+\lambda_{\mathrm{local}}\mathcal{L}_{\mathrm{local}}^{(a)}
\right).
$$

This provides 3D supervision through observable 2D distributions rather than through a volumetric target.

### 3.3 Plane-anchor conditioning

An anchor contains categorical labels, a plane normal, a coordinate on that axis, and an optional in-plane offset. Multiple anchors are assembled into a target tensor $Y$ and binary mask $M$, where $M(v)=1$ marks constrained voxel $v$. Full planes and smaller rectangular regions use the same representation.

The masked one-hot labels and mask are projected by a zero-initialized 3D convolution and added to the first denoiser feature map:

$$
h_A=\mathrm{Conv}_{3D}\!\left(
\left[\mathrm{onehot}(Y)\odot M,\;M\right]
\right).
$$

Zero initialization leaves the unconditioned mapping unchanged at the start of training. The anchor is provided at every reverse step, and constrained labels are optimized with masked cross-entropy,

$$
\mathcal{L}_{\mathrm{anchor}}
=-\frac{1}{|M|}\sum_{v:M(v)=1}
\log p_\theta\!\left(Y(v)\mid x_t,t,z,M\right).
$$

Correct labels alone do not ensure a compatible neighborhood. The 2D critics therefore also evaluate orthogonal sections centered where they cross an anchor, giving a seam-aware adversarial term $\mathcal{L}_{\mathrm{seam}}$. During training, anchor position and count are randomized, with up to four same-axis planes per sample.

### 3.4 Phase-fraction conditioning and full objective

An optional vector $v\in[0,1]^K$ specifies the desired phase fractions. Its embedding conditions the denoiser, while predicted mean fractions $\hat{p}$ receive

$$
\mathcal{L}_{\mathrm{vf}}=\lVert\hat{p}-v\rVert_1.
$$

The complete generator objective is

$$
\mathcal{L}_{G}=\mathcal{L}_{\mathrm{adv}}
+\lambda_{\mathrm{anchor}}\mathcal{L}_{\mathrm{anchor}}
+\lambda_{\mathrm{seam}}\mathcal{L}_{\mathrm{seam}}
+\lambda_{\mathrm{vf}}\mathcal{L}_{\mathrm{vf}}.
$$

The reported model uses 10 reverse transitions and weights $\lambda_{\mathrm{anchor}}=1$, $\lambda_{\mathrm{seam}}=0.25$, and $\lambda_{\mathrm{vf}}=1$. Condition dropout retains unconditioned sampling with the same network.

### 3.5 Shared-state tiled scale-up

Let $P$ be the block core size and $o$ the overlap on each side. Each $(P+2o)^3$ tile reads from one shared noisy volume. At reverse step $t$, a separable cosine-taper window $w_k$ fuses overlapping clean-volume predictions:

$$
\bar{x}_0(v)=
\frac{\sum_k w_k(v)\hat{x}_{0,k}(v)}
{\sum_k w_k(v)}.
$$

The global state is updated only after fusion, so adjacent tiles negotiate their overlaps throughout denoising rather than after generation. When a base volume is supplied, its inner core is retained exactly by construction. A cosine-weighted outer shell is only softly constrained, allowing the new surroundings to adapt to the base boundary.

## 4. Experimental Setup

### 4.1 Data and scope

The source is a 226 × 690 binary phase map. Phase 0 is pore and phase 1 is solid. Under an isotropy assumption, the same image distribution supervises all three axes. Random 128 × 128 crops are resized by nearest-neighbor sampling to 64 × 64 and augmented by rotations and reflections (Figure 1).

<p align="center">
  <img src="assets/paper/01-training-data.png" alt="Binary training image with three 128 by 128 crop regions" width="680">
</p>
<p align="center"><em>Figure 1. Binary 2D training image. Orange boxes show example 128 × 128 crops, which are resized to 64 × 64. Black denotes pore and gray denotes solid throughout.</em></p>

Evaluation uses 64 randomly sampled real crops. A fixed unconditioned 64³ sample from the trained generator serves as synthetic pseudo-ground truth for controlled anchor tests; it is denoted GT only in that context and is not experimental 3D ground truth. All real crops come from the training image, so the study is an in-sample proof of concept rather than a held-out generalization test.

### 4.2 Evaluation protocols

The single-plane test places one 64 × 64 crop at the axis-0 center of a 64³ volume. A fixed-seed coverage sweep then supplies 0, 1, 2, 4, 8, 16, 32, or 64 planes from the synthetic GT at evenly distributed axis-0 coordinates. The final multi-sample evaluation uses 25%, 50%, 75%, and 100% coverage with four random seeds.

For scale-up, a single-plane-anchored 64³ base is placed at the center of a 192³ output. The sampler uses 3 × 3 × 3 block cores with 16-voxel overlap. An eight-voxel shell on each base face may adapt, leaving a hard-retained 48³ base core.

### 4.3 Metrics

- **Kernel Inception Distance (KID):** the squared maximum mean discrepancy between 2,048-dimensional Inception features [16]. Table 1 compares 64 real crops with 64 generated axis-0 sections at the same 64 × 64 field of view; lower is better. For scale-up, one 64 × 64 crop is taken from each of 64 evenly spaced sections.

- **Porosity:** the fraction of pixels or voxels assigned to phase-0 pore. Agreement with the reference value is desired.

- **Tortuosity:** the axis-0 diffusive tortuosity factor of the pore phase, computed by a steady-state voxel diffusion solve [17]. It follows $D_{\mathrm{eff}}=D\varepsilon/\tau$, where $\varepsilon$ is porosity; agreement with the reference is desired.

- **Voxel accuracy:** the fraction of all 64³ voxels whose phase matches the synthetic GT at the same coordinate. This is a whole-volume recovery score, not accuracy restricted to supplied planes.

- **Local pore-continuation drop:** for adjacent planes normal to axis $a$, $C_a=P(X_{i+1}=0\mid X_i=0)$. The reported value is the three-axis mean $\Delta C=C_{\mathrm{interior}}-C_{\mathrm{boundary}}$, excluding pairs within four voxels of each boundary from the interior estimate. Zero indicates no measured boundary effect; a negative value means slightly greater local continuation at the boundary.

Table 1 reports mean values over four random seeds. The real-data KID is a baseline between independent crop sets. Figure 5 is a separate fixed-seed diagnostic whose KID is measured against synthetic-GT sections.

## 5. Results

Table 1 summarizes the quantitative results. The unconditioned 3D samples reproduce the controlled reference porosity and axis-0 tortuosity closely on average. Phase-fraction conditioning moves mean porosity slightly closer to the target, from 0.3499 to 0.3530 versus 0.3518, but does not improve KID. Thus, composition control and section-distribution similarity should be evaluated separately.

| Evaluation data | KID | Porosity | Tortuosity | Voxel accuracy | Local pore-continuation drop |
|---|---:|---:|---:|---:|---:|
| Controlled reference (GT) | — | 0.352 | 2.224 | — | — |
| Real 2D crops | 0.0025 | 0.3625 | — | — | — |
| 3D | 0.0282 | 0.3499 | 2.2337 | — | — |
| 3D (phase-fraction conditioned) | 0.0294 | 0.3530 | 2.1957 | — | — |
| 3D (anchored, 25%) | 0.0103 | 0.3431 | 2.4668 | 89.07% | — |
| 3D (anchored, 50%) | 0.0138 | 0.3286 | 2.2980 | 95.61% | — |
| 3D (anchored, 75%) | 0.0184 | 0.3347 | 2.2747 | 96.92% | — |
| 3D (anchored, 100%) | 0.0177 | 0.3392 | 2.2412 | 98.17% | — |
| 3D (scale-up) | 0.0269 | 0.3435 | 2.3263 | — | −0.68 pp |

### 5.1 Three-dimensional synthesis

The cutaway in Figure 2 exposes both the surface and interior of a generated 64³ volume. Its three orthogonal center sections (Figure 3) show feature scales comparable to those in the training image. These figures provide qualitative context; Table 1 supplies the distributional and transport measurements.

<p align="center">
  <img src="assets/paper/02-generated-volume.png" alt="Generated binary 3D volume with one octant removed" width="470">
</p>
<p align="center"><em>Figure 2. Generated 64³ volume with one octant removed. Black denotes pore and gray denotes solid.</em></p>

<p align="center">
  <img src="assets/paper/03-generated-slices.png" alt="Three orthogonal center sections of a generated volume" width="760">
</p>
<p align="center"><em>Figure 3. Center sections normal to axes 0, 1, and 2 of the volume in Figure 2.</em></p>

### 5.2 Plane anchoring

In the fixed-seed single-plane example, the generated center section matches all 4,096 supplied voxels without post-generation replacement (Figure 4). This is anchor-plane accuracy for one controlled example and is distinct from the whole-volume accuracy in Table 1.

<p align="center">
  <img src="assets/paper/04-anchor-conditioning.png" alt="Supplied center section, matching generated section, and anchored 3D volume" width="780">
</p>
<p align="center"><em>Figure 4. Fixed-seed single-plane conditioning. (a) Supplied axis-0 center section; (b) generated section, matching 4,096/4,096 constrained voxels; (c) surrounding 64³ volume.</em></p>

Across four seeds, whole-volume accuracy increases from 89.07 ± 0.10% at 25% coverage to 98.17 ± 0.06% at full coverage. The denser fixed-seed sweep in Figure 5 shows the same trend from the unanchored baseline: 54.93% at zero planes and 98.26% at all 64 planes. Porosity, tortuosity, and KID are not monotonic because they measure global or distributional properties rather than coordinate-wise identity.

<p align="center">
  <img src="assets/paper/06-anchor-sweep-metrics.png" alt="Whole-volume accuracy, KID, porosity, and tortuosity versus supplied axis-0 planes" width="780">
</p>
<p align="center"><em>Figure 5. Fixed-seed controlled-reference diagnostic for 0–64 supplied axis-0 planes. (a) Accuracy is measured over the complete 64³ volume. (b) KID compares generated sections with synthetic-GT sections and is not directly comparable with Table 1. Dashed lines in (c,d) show GT porosity and tortuosity. No across-volume uncertainty is shown because the seed is fixed.</em></p>

### 5.3 Anchored scale-up

The shared-state sampler expands the 64³ base to 192³. In the illustrated center plane, the complete 64 × 64 anchor retains 3,809 of 4,096 voxels (92.99%) because the outer shell may adapt, whereas all 2,304 voxels in the protected 48 × 48 anchor region remain exact. The complete 48³ base core is hard-retained by construction. Across four samples, the local pore-continuation drop at tile boundaries is −0.68 ± 0.13 percentage points, so this statistic detects no reduction relative to ordinary interior pairs.

<p align="center">
  <img src="assets/paper/05-scale-up.png" alt="Anchor, scaled center section, and cutaway of a 192 cubed volume" width="780">
</p>
<p align="center"><em>Figure 6. Scale-up with 3 × 3 × 3 block cores and 16-voxel overlap. (a) 64² center-plane anchor; (b) 192² center section, with the orange box marking the full 64² anchor extent and the blue dashed box the protected 48² region; (c) cutaway of the 192³ output.</em></p>

## 6. Discussion

Plane anchors change the task from statistical synthesis to coordinate-aware conditional synthesis. The controlled sweep shows that additional sections consistently narrow the set of admissible volumes: whole-volume accuracy rises monotonically even though KID does not. This distinction matters in practice. KID asks whether an unordered set of sections has a similar feature distribution, while voxel accuracy asks whether a specific structure appears at the correct location.

The morphology metrics reveal a second distinction. Porosity depends only on phase count and is relatively stable; tortuosity depends on connected transport paths and varies more under sparse anchoring. At full coverage, mean tortuosity approaches the controlled reference, but porosity remains lower. Anchors therefore provide strong spatial information without guaranteeing monotonic improvement in every aggregate property.

Joint tiled denoising produces a 27-fold increase in voxel count while preserving a chosen base core. The slightly negative local pore-continuation drop indicates no measured penalty at tile boundaries for the tested setting. It does not prove topological or permeability equivalence, nor isolate the effect of shared-state fusion, because independent-tile and overlap ablations were not performed.

## 7. Limitations

The synthetic GT is sampled from the trained model and therefore lies within its learned distribution. It tests whether anchor information can recover a known target, but not whether the method reconstructs unseen experimental 3D material. The real crop baseline, training data, and single-plane anchor also come from the same 2D image, with no held-out specimen.

The evaluation uses four samples, one binary image, an isotropy assumption, and axis-0 anchors and tortuosity. Multi-axis or intersecting anchors, anisotropic and multiphase media, and independent experimental volumes remain untested. KID features are learned from natural images and may miss material-specific morphology. Finally, local pore continuation measures only adjacent-voxel agreement; connected components, percolation, and permeability are needed for stronger physical validation.

## 8. Conclusion

This work integrates plane anchors and shared-state tiled diffusion into a 2D-supervised categorical 3D generator. Controlled experiments show that anchor coverage increases coordinate-wise recovery, while 3D tiled sampling expands the volume 27-fold and preserves a protected core without a measured reduction in local pore continuation at block boundaries. The method is a proof of concept for retaining measured 2D information inside large stochastic 3D microstructures; validation on held-out experimental volumes is the next requirement.

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

[10] G. Zhang et al., “Towards coherent image inpainting using denoising diffusion implicit models,” in *Proceedings of the 40th International Conference on Machine Learning*, vol. 202, pp. 41164–41193, 2023.

[11] A. Lugmayr et al., “RePaint: Inpainting using denoising diffusion probabilistic models,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 11461–11471, 2022.

[12] N. Hoffman, C. Diniz, D. Liu, T. M. Rodgers, A. Tran, and M. D. Fuge, “GrainPaint: A multi-scale diffusion-based generative model for microstructure reconstruction of large-scale objects,” *Acta Materialia*, vol. 288, art. 120784, 2025.

[13] Z. Ding, M. Zhang, J. Wu, and Z. Tu, “Patched denoising diffusion models for high-resolution image synthesis,” in *International Conference on Learning Representations*, 2024.

[14] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in *Advances in Neural Information Processing Systems*, vol. 33, pp. 6840–6851, 2020.

[15] Z. Xiao, K. Kreis, and A. Vahdat, “Tackling the generative learning trilemma with denoising diffusion GANs,” in *International Conference on Learning Representations*, 2022.

[16] M. Bińkowski, D. J. Sutherland, M. Arbel, and A. Gretton, “Demystifying MMD GANs,” in *International Conference on Learning Representations*, 2018.

[17] S. J. Cooper, A. Bertei, P. R. Shearing, J. A. Kilner, and N. P. Brandon, “TauFactor: An open-source application for calculating tortuosity factors from tomographic data,” *SoftwareX*, vol. 5, pp. 203–210, 2016.
