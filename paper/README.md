<div align="center">
  <h1>Generating 3D Microstructures from 2D Sections<br>with a Two-Stage Diffusion-GAN</h1>
</div>

## Abstract

We present a two-stage diffusion-GAN that learns 3D categorical microstructures from unregistered 2D sections: a low-resolution (LR) stage generates the 3D arrangement, and a super-resolution (SR) stage refines it to the source resolution. Both denoisers are trained only through 2D critics, with losses that penalize differences between the axes of each volume and, for LR, between mirrored diagonal statistics. Each stage's exported step is chosen by a section-structure score normalized by the real data's own sampling noise. On an isotropic binary image, LR volumes match the real pore fraction, percolate in both phases and show no preferred particle tilt, but fall 7% short of the real interface density, and their pore tortuosity is 9–11% higher along one axis than along the other two. SR sections lie 1.6–2.2 times closer to real sections than upsampled LR volumes while keeping the LR arrangement. A second training seed changes the section scores by 13% in LR and 36% in SR. Supplied planes are applied at inference by diffusion posterior sampling without anchor-specific training: a plane is reproduced with about 94% voxel agreement and no seam, and sixteen planes recover 89.6% of a known volume. On a simulated graphite–silicon anode with in-plane flakes and depth-graded silicon, depth-conditioned volumes follow the real depth profile and, against the real 3D volumes, reproduce the strongly anisotropic pore tortuosity within 1.2% and the flake thickness within 1%. They merge silicon particles into spanning bodies, which longer SR training removes, and give every volume the same flake tilt, whereas the two real volumes tilt in opposite directions.

## Introduction

Three-dimensional connectivity controls many transport and mechanical properties, but 3D scans are harder to obtain than 2D micrographs. Feature-matching and conditional neural methods reconstruct 3D microstructures from a 2D exemplar [1,2], and SliceGAN [3] learns a volumetric generator from 2D images alone by judging its sections with 2D critics. Diffusion models have since been used for 2D-to-3D expansion [4,5,6], property-conditioned methods control phase fraction or spatial statistics [7,8], and volumetrically supervised methods reconstruct from sparse or surface slices [9,10,11].

Two problems remain when only 2D sections exist. Section realism is a weak constraint on the volume: a generator can keep every section plausible while choosing an unrealistic 3D arrangement, for example by shearing structure so that only one section plane degrades. And without 3D data there is no direct way to decide which training step to export. Applications also need known planes, such as a measured surface, to be honored without training a conditional model; diffusion inpainting preserves observed regions [12,13], and diffusion posterior sampling (DPS) [17] steers an unconditional model with the gradient of a measurement error on its clean prediction.

Here a low-resolution (LR) stage generates the 3D arrangement and a super-resolution (SR) stage, trained the same way on frozen LR volumes, refines it to the source resolution. Each is a 3D denoising-diffusion GAN [14,15] trained only through 2D critics on pairs of adjacent diffusion states, with the relativistic loss and burn-in of R3GAN [16], and losses on the whole volume penalize differences between its three axes and between its mirrored diagonals, to discourage shear. The exported step is chosen by a section-structure score computed from 2D data, and known planes are applied at inference by DPS. We test it on an isotropic binary image and on a simulated anode whose flakes lie in one plane and whose composition changes with depth, where real 3D volumes allow direct comparison. The goal is plausible realizations from unregistered 2D sections, not a unique reconstruction.

## Results

### Two-stage generation from 2D sections

The training data are one 226 × 690 binary image (Figure 1). Random 128² crops supervise all three axes under an isotropy assumption. Each crop is area-downsampled by two for the LR stage, so one LR voxel spans two source pixels. The LR model thus sees a large field of view and learns the arrangement of particles, while the SR model, refining frozen LR volumes back to the source resolution, mostly adds boundary detail consistent with it. The LR stage trained for 10,000 steps (88 minutes on one Apple-silicon GPU) and the SR stage for 5000 steps (3.0 hours).

<p align="center">
  <img src="assets/01-training-data.png" alt="Binary training image with example crop regions" width="680">
</p>
<p align="center"><em>Figure 1. Binary training image. Orange boxes are example 128² crops; the middle one is the external anchor. Black: phase 0; gray: phase 1.</em></p>

### Training and checkpoint selection

The selection score falls quickly in the first 1500 steps and more slowly through burn-in, to 3.4 at step 5000; after burn-in it stays between 3.3 and 3.6 and is lowest at step 8500 (Figure 2a). Its components move differently: cluster error falls by a factor of 2.5 over the run, while two-point error, apart from a peak at steps 2000–2500, stays near its step-3000 level and drifts up by about 40% to the end (Figure 2b). The critic separates real from generated pairs increasingly well: the gap between their mean logits widens from about 1.9 at the end of burn-in to 2.6–2.8 from step 8000 on, mostly because real logits rise (Figure 2c). The score stops improving while this gap widens.

<p align="center"><img src="assets/lr-training.png" alt="Selection score, its components, and mean critic logits over LR training" width="900"></p>
<p align="center"><em>Figure 2. LR training. (a) Selection score; lower is better. (b) Its components, averaged over planes and divided by the half-split error of the evaluation crops (Table 1), so their scale differs from (a). (c) Mean critic logit of real and generated pairs, 200-step moving average. Gray: burn-in. Dashed: selected step 8500.</em></p>

On a second random set of crops (Table 1), the selected step 8500 again scores lowest of the five evaluated steps, 6% below the last step (scores here exceed selection scores, which use fewer crops; Methods). The pore fraction matches the real crops from step 5000 on, so the two-point error is not a phase-fraction offset: the real correlation shape at the generated pore fraction would give a two-point error of 0.0008, of the 0.0063 observed.

| Step | Structure ↓ | Two-point | Chord | Cluster | Pore fraction | Interface density |
|---|---:|---:|---:|---:|---:|---:|
| Real crops | — | — | — | — | 0.367 ± 0.050 | 0.1227 ± 0.0071 |
| 2500 | 6.20 | 0.0120 | 0.0870 | 0.434 | 0.380 ± 0.005 | 0.1127 ± 0.0010 |
| 5000 | 4.87 | 0.0049 | 0.0876 | 0.342 | 0.367 ± 0.007 | 0.1122 ± 0.0013 |
| 7500 | 4.66 | 0.0061 | 0.0891 | 0.309 | 0.367 ± 0.007 | 0.1146 ± 0.0019 |
| **8500 (selected)** | **4.50** | 0.0063 | 0.0874 | 0.293 | 0.367 ± 0.007 | 0.1144 ± 0.0018 |
| 10000 (last) | 4.81 | 0.0067 | 0.0935 | 0.313 | 0.370 ± 0.006 | 0.1136 ± 0.0017 |

<p align="center"><em>Table 1. Section errors of 16 volumes per step against a second random set of 128 real crops, averaged over the three planes. ± is one standard deviation, over crops for real values and over volumes for generated ones.</em></p>

### Microstructure of generated volumes

The sections of the selected step reproduce rounded solid particles separated by a connected pore network (Figures 3 and 4), but the interface density is 6.8% below the real crops', whose own standard deviation across crops is 5.8% of the mean. Both phases percolate along every axis (99.9% of pore and 98.0% of solid voxels lie in spanning components). No measured 3D volume exists, so the 3D measures are judged against the isotropic, mirror-symmetric structure assumed in training. The mirror symmetry is met: solid particles show no preferred tilt (balances −0.01, −0.07 and 0.03 in the zy, zx and yx planes over 194–198 particles, where a mirror-symmetric structure gives 0; every mean slope lies within 0.4 standard errors of 0). The isotropy is met only in part: pore tortuosity is 2.09 ± 0.06, 2.27 ± 0.18, and 2.04 ± 0.08 along z, y, and x, so z and x agree within 3% while y is 9–11% higher and varies twice as much between volumes.

<p align="center"><img src="assets/02-generated-volume.png" alt="Generated LR 64-cubed volume with one octant removed" width="470"></p>
<p align="center"><em>Figure 3. Unconditional 64³ LR sample at step 8500 with one octant removed.</em></p>

<p align="center"><img src="assets/03-generated-slices.png" alt="Three real LR crops and the three orthogonal center sections of a generated volume" width="700"></p>
<p align="center"><em>Figure 4. Top: real crops on the LR grid. Bottom: center sections of the volume in Figure 3.</em></p>

### Plane anchors

DPS raises agreement on the anchored plane from 53–55%, the chance agreement of unrelated sections at this pore fraction, to 94.3% for internal planes and 93.6% for the real external section (Table 2, Figure 5a–c). The external section is a real image rather than a model sample, yet it is matched nearly as well, also at the first section of the output (94.1%), which has generated structure on both sides because generation samples an 8-voxel margin around the output. Pool4 similarity is less than one point above exact agreement, so the remaining errors are not small boundary shifts but particles or necks placed differently.

| Anchor | Cases | Exact, no guidance | Exact, DPS ↑ | Pool4, DPS ↑ | First change, DPS | First change, baseline | Flicker / baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| Internal | 24 | 53.2 ± 3.6% | 94.3 ± 1.2% | 94.9 ± 1.1% | 0.64 ± 0.11 | 0.99 ± 0.13 | 1.68 ± 0.77 |
| External, center | 8 | 53.1 ± 2.9% | 93.6 ± 0.8% | 94.5 ± 0.9% | 0.59 ± 0.07 | 0.97 ± 0.12 | 1.73 ± 0.65 |
| External, boundary | 8 | 54.9 ± 2.8% | 94.1 ± 0.4% | 95.0 ± 0.4% | 0.61 ± 0.07 | 1.01 ± 0.16 | 1.16 ± 0.21 |

<p align="center"><em>Table 2. Single-plane anchors at the selected step.</em></p>

The anchor creates no seam: next to every kind of anchor, sections change less than ordinary adjacent sections (first change 0.59–0.64 against 0.97–1.01 for the baseline), consistent with DPS pulling the neighbors toward a structure compatible with the anchor. Flicker rises by a sixth to three-quarters, with a wide spread between cases, so the change rate varies more along the volume.

The anchor's 3D information reaches a few voxels (Figure 5d). Agreement with the internal reference falls from 94.3% on the plane to 85.0% one voxel away, 62.0% at four, and 57.2% at twelve, approaching the 53–54% of unrelated volumes. The far field, however, is not the baseline volume, because the gradient moves the whole noisy state (Methods).

<p align="center"><img src="assets/lr-anchor.png" alt="External anchor example and agreement curves for internal anchors and nested anchors" width="900"></p>
<p align="center"><em>Figure 5. Plane anchors. (a) Real external section; (b) generated plane at the anchored coordinate; (c) the same coordinate without guidance; right: sections at distance d from the anchor. (d) Agreement with a known reference against distance from one anchored plane; shading: ±1 SD over the 24 cases. (e) Agreement with a known reference as nested planes are anchored.</em></p>

More planes recover more of a known volume (Figure 5e): agreement rises from 54.1% with no anchors to 77.7% with eight planes and 89.6% with sixteen, and unanchored planes follow closely, so the gain is not only the anchored planes themselves. Beyond sixteen planes the curve levels off near 93%, slightly below single-plane accuracy; with all 64 planes anchored, the anchored planes themselves agree to 91.8%.

### Super-resolution

The SR selection score rose slightly over the first 1500 steps, then fell at every evaluation from step 2000 on and was lowest at the last step, 5000 (Figure 6a), which also beats step 2500, the only other saved step, on the second crop set (Table 3). As in LR, a longer run might score lower; the anode below tests this. Real and generated critic logits fall together, about 0.3–0.4 apart throughout (Figure 6c), so the widening gap seen in LR does not appear within these steps. As the score falls, the downsampled output moves away from its LR input (Figure 6b): SR improves its sections partly by changing the coarse structure it is given.

<p align="center"><img src="assets/sr-training.png" alt="SR selection score, mismatch with the LR input, and mean critic logits over SR training" width="900"></p>
<p align="center"><em>Figure 6. SR training. (a) Selection score at source resolution; lower is better. (b) Mean squared difference between the area-downsampled output and the LR input on the selection inputs. (c) Mean critic logit of real and generated pairs, 200-step moving average. Gray: burn-in. Dashed: selected step 5000.</em></p>

At source resolution, SR sections score well below nearest-neighbour and trilinear upsampling of the same LR volumes (Table 3, Figure 7). Nearest upsampling keeps the two-voxel steps of the LR grid, which the chord distribution penalizes; trilinear upsampling smooths the steps but raises the cluster error. SR lowers the chord error to less than a fifth of nearest upsampling's and the cluster error slightly below it. Like both upsamplings, it shows about two-fifths fewer separate pore regions per section than real crops, a deficit of its LR inputs, and it splits the solid into 20% more regions than real crops (nearest: 4% fewer), visible as small specks in Figure 7. The interface density falls 2.6% short of the real crops', whereas the upsamplings fall 11% and 13% short. SR sections are not indistinguishable from real ones: chord and cluster errors remain about four and ten times the real half-split error.

| Volumes | Structure ↓ | Two-point | Chord | Cluster | Pore fraction | Interface density | LR agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| Real crops | — | — | — | — | 0.368 ± 0.050 | 0.0635 ± 0.0043 | — |
| LR, nearest | 11.19 | 0.0069 | 0.5004 | 0.349 | 0.367 ± 0.007 | 0.0567 ± 0.0009 | 100% |
| LR, trilinear | 8.37 | 0.0074 | 0.1574 | 0.524 | 0.366 ± 0.007 | 0.0553 ± 0.0008 | 99.1% |
| SR, step 2500 | 8.14 | 0.0061 | 0.0975 | 0.587 | 0.367 ± 0.007 | 0.0595 ± 0.0009 | 98.8% |
| **SR, step 5000 (selected)** | **5.19** | 0.0069 | 0.0901 | 0.321 | 0.364 ± 0.008 | 0.0618 ± 0.0011 | 98.5% |

<p align="center"><em>Table 3. Section errors at source resolution of 16 HR volumes, all refined from the LR volumes of Table 1's selected step, against the crops of Table 1 at source resolution; averages and ± as in Table 1.</em></p>

<p align="center"><img src="assets/sr-sections.png" alt="Real crops at source resolution and center sections of one LR volume upsampled by nearest and trilinear interpolation and refined by SR" width="640"></p>
<p align="center"><em>Figure 7. Top: real crops at source resolution. Below: center sections of one LR volume after nearest-neighbour and trilinear upsampling, and after SR.</em></p>

SR keeps the arrangement chosen by LR. Downsampled back to the LR grid, 98.5% of LR voxels keep their LR phase, and SR changes 5.1% of HR voxels relative to nearest upsampling. Two SR seeds on the same LR input differ in 3.2% of voxels, so part of what SR adds is set by its LR input and part varies between samples. Both phases still percolate along every axis (99.9% of pore and 98.2% of solid voxels lie in spanning components). Pore tortuosity is 1.89 ± 0.05, 1.98 ± 0.13, and 1.81 ± 0.06 along z, y, and x, 3–6% below nearest-upsampled LR volumes on the same grid (1.96, 2.11, 1.91) at an unchanged pore fraction, and keeps the higher value along y (Figure 8).

<p align="center"><img src="assets/sr-volume.png" alt="Generated HR 128-cubed volume with one octant removed" width="470"></p>
<p align="center"><em>Figure 8. SR refinement of the volume in Figure 3 to 128³ voxels, one voxel per source pixel, with one octant removed.</em></p>

### Run-to-run spread

The runs above are one training run per stage. Retraining with another training seed and the same settings gives a different score: a second LR run scores 3.92 against 4.50, and a second SR run on the same LR volumes scores 3.33 against 5.19 (trilinear upsampling of these volumes: 8.37). The second LR run also passes the 3D checks above (interface density 3.5% below real, lowest percolation 0.977, internal anchors 95.6%). SR varies more although its input is fixed, so a single SR run fixes the reported SR numbers only to within about a third.

### An anisotropic, depth-graded structure

The second test case is a simulated graphite–silicon anode for which real 3D volumes exist (Figure 9). Graphite flakes are oblate and lie roughly in the xy plane, and silicon spheres fill part of the pore space with a fraction that rises from 0 at z = 0 to about 0.3 at the far face. Sections of two simulated 256³ volumes are the training data; the volumes themselves are never shown to the model and serve as ground truth. Separate critics judge xy and side sections, both stages are conditioned on depth, and LR also on the phase-fraction profile along depth (Methods). The LR stage trained for 10,000 steps, with step 8500 selected, and the SR stage for 5000 steps, with step 4500 selected. Generated volumes are compared with the real volumes on the LR grid and at source resolution (Methods).

<p align="center"><img src="assets/anode-volume.png" alt="Real and generated 256-cubed anode volumes with depth drawn upward and one octant removed" width="800"></p>
<p align="center"><em>Figure 9. A real anode volume and an SR volume at source resolution, depth z drawn upward, with one octant removed. Black: pore; gray: graphite; light: silicon.</em></p>

Both stages follow the real depth profile (Figure 10). LR receives this profile as its condition, so its agreement shows that the condition is honored, not that the profile is learned. It falls short at the deep end: in the deepest eighth of the depth, LR volumes hold 0.203 silicon against 0.255 for the real volumes on the LR grid, and more graphite (0.299 against 0.266). SR, which receives the depth but no profile, recovers part of this silicon (0.221 against 0.261) while keeping 96.8% of LR voxels in their LR phase; its overall silicon fraction is 0.072 against 0.078.

<p align="center"><img src="assets/anode-profile.png" alt="Pore, graphite and silicon fractions against depth for real, LR and SR volumes" width="900"></p>
<p align="center"><em>Figure 10. Phase fraction of each xy section against depth. Real: mean of the two real volumes at source resolution; LR and SR: mean over generated volumes, shading ±1 SD.</em></p>

Sections are compared with the real sections at matching depths, and each error is divided by the error between the sections of the two real volumes, a coarse scale because these volumes differ in graphite fraction (0.395 and 0.365). On this scale LR scores 1.04 in xy and 0.87 on the side planes: its two-point and chord errors lie below the difference between the two real volumes, its cluster error 1.5–2.1 times this difference. SR scores 1.67 at source resolution, against 2.14 and 1.96 for nearest and trilinear upsampling, a smaller gain over upsampling than for the isotropic sample. The sections show what these averages hide (Figure 11): generated silicon forms irregular, partly elongated and merged bodies instead of separate spheres, and generated flakes are visibly tilted in yz sections.

<p align="center"><img src="assets/anode-sections.png" alt="Real, real on the LR grid, LR and SR anode sections in xy at two depths and in xz and yz" width="900"></p>
<p align="center"><em>Figure 11. Anode sections: xy at two depths (in source pixels) and the center xz and yz sections, of one real volume at source resolution and on the LR grid, and of one LR volume and its SR refinement. Area downsampling leaves exact ties at flake faces, broken at random, which speckles real xy sections on the LR grid.</em></p>

The real volumes allow direct 3D comparison (Table 4). Flakes lying in xy make pore transport much harder along z than along y or x, and the generated volumes reproduce this anisotropy: their pore tortuosity differs from the real values by at most 3.2% on the LR grid and 1.2% at source resolution, where the two real volumes themselves differ by about 10% along z. The flake thickness, the mean graphite chord along z, is within 1% of real on both grids, and lateral graphite chords within 7%. Silicon chords are 8–19% too long on the LR grid but within 4% at source resolution. Silicon connectivity, however, is wrong: in SR volumes 61% of the silicon lies in one component that spans y and x, whereas in the real volumes the largest component holds 5–6% and none spans an axis. As in the real volumes, pore percolates along every axis and graphite does not span the depth, but graphite also spans laterally less often (4–9% of graphite voxels against 30%).

| Measure | LR | Real, LR grid | SR | Real |
|---|---:|---:|---:|---:|
| Pore tortuosity, z | 2.16 ± 0.01 | 2.23 | 2.01 ± 0.01 | 2.01 |
| Pore tortuosity, y | 1.45 ± 0.01 | 1.46 | 1.32 ± 0.01 | 1.31 |
| Pore tortuosity, x | 1.36 ± 0.01 | 1.36 | 1.26 ± 0.01 | 1.24 |
| Graphite chord, z | 4.40 ± 0.03 | 4.37 | 8.52 ± 0.05 | 8.46 |
| Graphite chord, y | 8.42 ± 0.09 | 7.86 | 17.15 ± 0.27 | 17.38 |
| Graphite chord, x | 9.92 ± 0.09 | 9.34 | 20.67 ± 0.27 | 21.29 |
| Silicon chord, z | 4.55 ± 0.04 | 3.84 | 8.35 ± 0.03 | 8.06 |
| Silicon chord, y | 4.26 ± 0.05 | 3.96 | 8.28 ± 0.08 | 8.31 |
| Silicon chord, x | 4.40 ± 0.05 | 3.96 | 8.38 ± 0.02 | 8.30 |
| Silicon in largest component | 0.59 | 0.42, 0.15 | 0.61 | 0.06, 0.05 |
| Graphite tilt balance, dz/dy | 0.35 to 0.51 | −0.10, 0.69 | 0.46 to 0.55 | −0.06, 0.70 |
| Graphite tilt balance, dz/dx | −0.42 to −0.26 | 0.65, −0.56 | −0.36 to −0.30 | 0.66, −0.51 |
| Graphite tilt balance, dy/dx | −0.20 to −0.11 | −0.61, −0.40 | −0.21 to −0.12 | −0.68, −0.36 |

<p align="center"><em>Table 4. 3D measures of 8 LR and 4 SR volumes against the two real volumes on the same grid; chords are mean run lengths in voxels of that grid. ± is one standard deviation over generated volumes; real values are means of the two volumes, except the last four rows, which give each real volume. Tilt balances of generated volumes are the range over volumes.</em></p>

### Silicon particles and flake tilt

Two follow-up experiments probe the anode's errors. For silicon, the real volumes, area-downsampled to the LR grid, were refined by the SR model. The largest silicon component then holds 21–54% of the silicon, more than on the LR grid itself (42% and 15%), so SR does not separate spheres that the LR grid merges, and the LR inputs are not the only cause. Continuing the same SR run from step 5000 to 10,000, with no other change, separates them: on the same four LR volumes, the largest component falls from 61% to 19% at step 7500 and to 7.5% at step 10,000, against 5–6% real; silicon no longer spans y or x, the section score falls from 1.66 to 1.56, and tortuosity and flake thickness stay within 1.2% of real. This is one run, and the results above keep the 5000-step setting.

For the flake tilt, the two real volumes tilt in opposite directions (Table 4), because the simulation draws each volume's flake rotations anew; the pooled real tilt therefore describes neither volume. Every generated volume tilts like the second real volume and more weakly. A second LR training seed gives the same direction in all eight volumes, and so does judging xz and yz sections with separate critics, in two seeds; separate critics only weaken the tilt (dz/dx −0.05 to −0.23). A 2D critic judges one section at a time, drawn from both real volumes alike, so nothing asks one volume to keep one tilt throughout, and the generator settles on a single direction.

## Discussion

A 3D generator trained only through 2D critics can produce volumes whose sections lie within a few times the data's own sampling noise of real sections. Section statistics, however, do not pin down the third dimension. On the isotropic image, where no 3D data exist, the volumes meet the assumed mirror symmetry in particle tilt but not fully the assumed isotropy in tortuosity. On the anode, the real volumes show that the model reproduces the in-plane flakes and with them the anisotropic pore tortuosity, but merges the sparse silicon particles unless SR trains longer, and cannot represent a tilt that differs from one real volume to the other.

Selection by section structure needs only 2D data. On the isotropic image it chose LR step 8500, which also scored lowest on the second crop set, 6% below the last step, and confirmed the last SR step, whose score had not stopped falling; on the anode it chose earlier steps than the last in both stages. Its statistics can disagree, as the different trends of cluster and two-point error show, and their equal weighting is a choice, not a derivation, whose scale depends on the number of real crops. Retraining with another seed moved the scores by 13% in LR and 36% in SR, so differences between settings smaller than this cannot be judged from one run.

The two stages divide the errors. SR repairs what the LR grid cannot represent, the two-voxel steps that dominate the chord error, but it inherits the 3D arrangement and, on the isotropic image, the missing pore regions, so these have to be fixed in LR. It is not limited to boundaries, however: on the anode it partly restores the deep silicon that LR undershoots and, trained longer, separates silicon particles that LR merges. SR drifts from its input as its sections improve, so the consistency loss bounds rather than prevents changes to the coarse structure.

Posterior sampling avoids training a separate anchor model, at the price that the whole volume changes: DPS gives up a fixed baseline far from the anchor in exchange for continuity near it; pasting the plane into a fixed sample was not compared. That controlled recovery levels off slightly below single-plane accuracy suggests that DPS accuracy, not missing information, limits agreement; stronger guidance, or guidance during SR, may raise it.

Conditioning on depth lets one model generate a graded structure, and the profile condition sets its composition. A property that varies between specimens rather than with depth, such as the anode's net flake tilt, is not reached this way: the model would need a per-specimen condition or a critic that compares whole volumes.

The study has clear limits. The isotropic image has no measured 3D volume, so 3D reconstruction accuracy and true tortuosity cannot be measured there; its internal anchors come from the model itself, the external anchor is a crop of the training image, and its volumes are small relative to the particle size, so tortuosity, which depends on the grid, and percolation compare generated volumes on one grid rather than estimate material properties. The anode is simulated, and its real volumes also supply its training sections, so the 3D comparison tests 2D-to-3D generation, not generalization to an unseen specimen; with two real volumes, the real values and the section scale rest on two samples. The reported results come from one training run per stage and sample with no ablations, so they describe the integrated system, not the contribution of each part; the second isotropic runs and the anode experiments add single runs. The causes of the higher tortuosity along y and of the LR silicon deficit at depth are untested. Only the selected and four archived LR steps and two SR steps of the isotropic runs kept their weights, so the second crop set checks selection among those alone, and not independently: both crop sets come from the one image, and 8 of the 16 evaluated LR volumes use the selection's seeds. Isotropic LR volumes are generated without a target phase fraction, a condition met in only 10% of training steps and one on which the phase-fraction loss never acts (Methods). Isotropic volumes are evaluated as single 64³ blocks refined by SR in one tile; tiled generation and refinement are evaluated only on the anode, and DPS across tiles and on the anode not at all.

## Methods

### Data

In the isotropic image, phase 0 is pore and phase 1 is solid. The image's provenance and physical pixel size are unknown, so results are in voxels. Sections normal to each axis need not be registered; here one image serves all three.

The anode volumes come from a packing simulation in the repository. Graphite flakes are oblate ellipsoids drawn, for each volume, from a library of 48 rotated shapes whose normals scatter around z; silicon spheres then fill free space while each depth slice is below its target fraction. Phase 0 is pore, 1 graphite and 2 silicon. Sections normal to each axis of both volumes are the training images, so xz and yz sections span the full depth.

### Denoiser and pair critics

The forward process is a variance-preserving diffusion on one-hot phases scaled to ±1, with ten transitions. A 3D U-Net receives the noisy state, the transition index, a fresh latent vector, and a target phase fraction taken from a real crop, and predicts phase logits; with their softmax $p$, the clean prediction $\hat x_0=2p-1$ defines one Gaussian posterior step, as in denoising diffusion GANs [15]. Training drops the target fraction in 10% of steps for classifier-free guidance. The isotropic volumes reported here are generated without a target, so guidance does not change them; anode LR volumes are generated with their depth profile as target (Depth conditioning) and the repository's default guidance.

Each training step generates one volume up to a random transition $t$ and cuts sections normal to every axis from the pair $(x_t,x_{t+1})$. A 2D critic scores the two sections together, $D(x_t,x_{t+1},t)$, at each scale of an area pyramid down to 16² pixels, three scales for LR; real pairs come from noising real crops. For an isotropic material one critic serves all planes, and every real and generated pair receives a random flip or 90° rotation other than the identity. The critic is trained with the relativistic pairing loss of R3GAN [16],

$$
\mathcal{L}_D=\mathbb{E}\big[\log(1+e^{D(\tilde x)-D(x)})\big],
$$

with each generated pair $\tilde x$ matched to a real pair $x$, plus zero-centered gradient penalties on real (R1) and generated (R2) pairs, taken with respect to the less noisy section $x_t$ only and applied every eighth critic update. R3GAN motivates this regularized relativistic loss as a remedy for mode dropping and non-convergence [16]. Over the first half of training, a cosine burn-in lowers the learning rates to a quarter and the penalty weights to a fraction of their values, raises Adam's $\beta_2$ from 0.9 to 0.99, and lengthens the EMA [16].

### Statistical losses on the clean volume

Critics judge sections one at a time. Four further terms act on the predicted phase probabilities of the volume. Real transitions and axis agreement act only at the last transition ($t=0$), each with weight 0.1 ramped in over the burn-in.

- **Phase fraction** matches the volume's mean phase fractions to the target fraction, when one is given, by KL divergence.
- **Real transitions** match the joint distribution of phases at in-plane gaps of up to 8 voxels, on 16 sections per plane, to that of real crops, a direct signal on correlation length.
- **Axis agreement** penalizes differences between the phase-pair statistics along the axes normal to the planes one critic judges, three for an isotropic material, at gaps of 1–8 voxels, beyond a tolerance of 0.01. Unlike the critic, it compares directions inside one volume, so anisotropy is penalized even when every plane looks realistic.
- **Mirror balance**, used for the isotropic image, targets shear. A structure such as $f(y,x-z)$ keeps xy and yz sections realistic and turns only xz sections into stripes, which axis-aligned statistics cannot see. For each coordinate plane and diagonal offset $(a,b)$, the loss compares the phase-pair table $P$ at $(+a,+b)$ with its mirror at $(+a,-b)$:

$$
\mathcal{L}_{\mathrm{mirror}}=\mathbb{E}_{\text{plane},(a,b)}\Big[\max\big(0,\;\tfrac12\lVert P_{+a,+b}-P_{+a,-b}\rVert_1-\epsilon\big)\Big],
$$

with $\epsilon=0.02$ and weight 10, at every transition.

### Depth conditioning

For the anode, one critic judges xy sections and a second judges xz and yz sections, axis agreement compares only y and x, and augmentations that would reverse z or exchange it with y or x are disabled. Both denoisers receive each voxel's depth, normalized over the source depth. The LR denoiser also receives a target phase-fraction profile along depth, with its mean as the target fraction: in training, the profile of the real side section that sets the volume's depth range, which a loss matches to the volume's mean fractions per depth; in generation, the profile of all training sections. Larger volumes are generated in overlapping tiles, each with the depth and profile of its own position.

### Super-resolution

The SR denoiser has the same form, with the LR phase fractions, resized to the output grid, as an extra input. It refines volumes from the frozen LR generator, held in a bank of eight volumes of which one is replaced every 500 steps, and is judged by its own pair critic of the same design on real crops at source resolution (four pyramid scales), together with axis agreement. SR's axis agreement adds the LR input's own axis differences to the tolerance, so it does not ask SR to remove anisotropy that LR generated. Random coarse blocks of the input are sometimes replaced by random phases, and the network receives the corrupted fraction, so it is not forced to copy errors in its input. In place of the other statistical losses, a consistency loss with weight 10 penalizes the squared difference between the area-downsampled clean prediction and the LR input on uncorrupted voxels beyond a tolerance of 0.005.

### Checkpoint selection

Every 500 steps the EMA generator produces 8 volumes from fixed seeds, and 128 of their sections per plane are compared with 64 fixed real crops per plane by three statistics: two-point correlation, chord-length distribution, and cluster density and size per phase. Cluster statistics are included because merged pores and fragmented solid can match both two-point and chord statistics while changing 3D connectivity. Each error is divided by the same error between two halves of the real crops, so 1 means "as far from the data as the data is from itself" at this number of crops; the normalized errors are averaged over statistics and planes into one score, and the lowest-scoring step is exported. The half-split error shrinks as crops are added, so the 128-crop evaluation below gives larger scores than selection (4.50 against 3.31 for step 8500). SR is scored on refinements of fixed bank volumes at source resolution. With depth conditioning, scored volumes take the depth ranges of fixed real side crops.

### Plane anchors

The denoiser is trained without any anchor condition. At inference, an anchor with values $y$ on mask $M$ steers every reverse step but the last by DPS [17]:

$$
x_t\leftarrow x_t-\frac{s}{\mathrm{RMS}}\,\nabla_{x_{t+1}}\big\lVert M\odot(\hat x_0(x_{t+1})-y)\big\rVert^2,
$$

where RMS is the root-mean-square anchor error, so the step does not depend on how much anchor area a volume holds. The gradient flows through the whole denoiser and moves the entire noisy state, so neighboring sections are pulled toward a structure compatible with the anchor instead of meeting it at a seam. Nothing is clamped or pasted; the anchored plane itself is generated.

### Evaluation

**Structure.** Each evaluated checkpoint generates 16 unconditional 64³ volumes from seeds 0–15, which include the selection's seeds 0–7, and compares them with a second random set of 128 crops of the training image, drawn with another seed than the selection's; the same crops serve all three planes. The structure score is computed as in selection, with the half-split error of these 128 crops. Real values are averages over random crops, the distribution the model is trained on. *Interface density* is the fraction of neighboring voxel pairs with different phases, averaged over the axes of a section or volume. For the selected step we also report percolation (fraction of a phase in components spanning an axis), pore tortuosity along each axis [18], and the tilt balance of solid particles (fraction tilted one way minus the other).

**Super-resolution.** The 16 LR volumes of the selected LR step are refined once each, in one tile, and compared with the same 128 crops at source resolution; the baselines upsample the same LR phase fractions. *LR agreement* is the fraction of LR voxels whose majority phase after area downsampling of the HR volume is the LR phase. Tortuosity and percolation are computed as for LR.

**Anode.** Eight LR volumes of 128³ voxels, seeds 0–7, assembled from overlapping 64³ tiles to cover the full width and depth of the real volumes, are generated, and the first four are refined by SR to 256³ in tiles. LR volumes are compared with the real volumes area-downsampled to the LR grid, with exact ties broken at random, and SR volumes with the real volumes at source resolution. Section statistics compare 128 evenly spaced sections per plane of each set, so every depth is represented, and are scaled by the same errors between the sections of the two real volumes. Tortuosity, percolation, mean chord length (the mean run length of a phase along an axis) and tilt balance are computed on whole volumes.

**Anchors.** Each anchored sample is compared with a same-noise baseline, the same seed without guidance. Guidance moves the whole noisy state, so anchored and baseline volumes differ in 36–47% of voxels at every distance from the anchor, about as much as unrelated volumes (about 46%); the baseline is a reference sample of the model, not a paired counterfactual. *Internal anchors* take the center plane of an unconditional reference volume along each axis (24 cases); since the reference is known in 3D, agreement can be measured at every distance from the plane. *External anchors* use the real crop of Figure 1 at the center or the first section of the 64³ output (8 cases each). *Controlled recovery* anchors a growing, nested set of planes from a reference volume (4 cases). *Pool4* is the agreement of 4 × 4-cell phase fractions, which tolerates small boundary shifts. *First change* is the label-change rate next to the anchor divided by the ordinary rate between adjacent sections; above 1 indicates a seam. *Flicker* is the 95th percentile of the change in adjacent-section change rate, relative to the baseline.

## Code availability

Code, settings, the training image, the anode simulation, and the evaluation scripts are available at https://github.com/phykn/diffusion-gan3d. The reported runs used the `default` and `simul_anode` samples of v2 (commit cb99772); the second runs differ only in `train.seed`. `paper/scripts/evaluate_lr.py --run <LR run>`, `evaluate_sr.py --run <SR run>` and `evaluate_anode.py --run <anode SR run>` write the reported numbers to `paper/assets/lr-results.json`, `sr-results.json` and `anode-results.json`, and `make_figures.py` draws the figures from them. The anode follow-ups resumed the SR run with `train.py --resume <checkpoint> --steps 10000` and set `tune.lr.model.critic.plane_groups: [[xy], [xz], [yz]]`; their numbers are not in `paper/assets/`. The v1 results are under the `v1` tag.

## References

[1] R. Bostanabad, “Reconstruction of 3D microstructures from 2D images via transfer learning,” *Computer-Aided Design*, vol. 128, art. 102906, 2020.

[2] J. Feng, Q. Teng, B. Li, X. He, H. Chen, and Y. Li, “An end-to-end three-dimensional reconstruction framework of porous media from a single two-dimensional image based on deep learning,” *Computer Methods in Applied Mechanics and Engineering*, vol. 368, art. 113043, 2020.

[3] S. Kench and S. J. Cooper, “Generating three-dimensional structures from a two-dimensional slice with generative adversarial network-based dimensionality expansion,” *Nature Machine Intelligence*, vol. 3, pp. 299–305, 2021.

[4] J. Phan et al., “Generating 3D images of material microstructures from a single 2D image: a denoising diffusion approach,” *Scientific Reports*, vol. 14, art. 6498, 2024.

[5] K.-H. Lee and G. J. Yun, “Multi-plane denoising diffusion-based dimensionality expansion for 2D-to-3D reconstruction of microstructures with harmonized sampling,” *npj Computational Materials*, vol. 10, art. 99, 2024.

[6] C. Düreth, P. Seibert, D. Rücker, S. Handford, M. Kästner, and M. Gude, “Conditional diffusion-based microstructure reconstruction,” *Materials Today Communications*, vol. 35, art. 105608, 2023.

[7] A. Sadeghkhani, B. Bennett, and A. Rabbani, “Property-constrained 3D porous media reconstruction from 2D images via conditional generative adversarial networks,” in *87th EAGE Annual Conference & Exhibition*, vol. 2026, pp. 1–5, 2026.

[8] Z. Ma et al., “A sliced-Wasserstein and neural network framework for statistically controllable 3D microstructure reconstruction,” *Computer-Aided Design*, vol. 198, art. 104100, 2026.

[9] S. Fan, Y. Li, X. Wang, and D. Du, “Dual-domain latent diffusion framework with multi-modal conditioning for controllable porous media reconstruction,” *Computational Materials Science*, vol. 272, art. 114876, 2026.

[10] Y. Shi et al., “GeoTopoDiff: Learning geometry–topology graph priors through boundary-constrained mixed diffusion for sparse-slice 3D porous reconstruction,” *arXiv preprint* arXiv:2605.03764, 2026.

[11] Y. Meng, J. Jiang, B. Zhao, S. Ye, D. Wang, and J. Wu, “Inferring internal structures of porous media from orthogonal external surfaces: A novel 2D-to-3D reconstruction method based on a deep conditional diffusion model,” *Journal of Geophysical Research: Machine Learning and Computation*, vol. 3, art. e2025JH001101, 2026.

[12] G. Zhang et al., “Towards coherent image inpainting using denoising diffusion implicit models,” in *Proceedings of the 40th International Conference on Machine Learning*, pp. 41164–41193, 2023.

[13] A. Lugmayr et al., “RePaint: Inpainting using denoising diffusion probabilistic models,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 11461–11471, 2022.

[14] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in *Advances in Neural Information Processing Systems*, vol. 33, pp. 6840–6851, 2020.

[15] Z. Xiao, K. Kreis, and A. Vahdat, “Tackling the generative learning trilemma with denoising diffusion GANs,” in *International Conference on Learning Representations*, 2022.

[16] Y. Huang, A. Gokaslan, V. Kuleshov, and J. Tompkin, “The GAN is dead; long live the GAN! A modern GAN baseline,” in *Advances in Neural Information Processing Systems*, vol. 37, 2024.

[17] H. Chung, J. Kim, M. T. McCann, M. L. Klasky, and J. C. Ye, “Diffusion posterior sampling for general noisy inverse problems,” in *International Conference on Learning Representations*, 2023.

[18] S. J. Cooper, A. Bertei, P. R. Shearing, J. A. Kilner, and N. P. Brandon, “TauFactor: An open-source application for calculating tortuosity factors from tomographic data,” *SoftwareX*, vol. 5, pp. 203–210, 2016.
