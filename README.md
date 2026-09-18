# Self-Supervised-Pretraining-and-Fine-Tuning-Model-for-Medical-Image-Segmentation-
# Self-Supervised MAE Pretraining + ResUNet Fine-Tuning for Brain Tumour Segmentation

A two-stage semi-supervised pipeline that uses **Masked Autoencoder (MAE)** pretraining on unlabelled MRI volumes to boost **ResUNet** segmentation performance on the BraTS brain tumour dataset — particularly in low-label-data regimes.

## Motivation

Supervised medical image segmentation models need large volumes of expert-annotated data, which is expensive and slow to produce, especially for rare conditions. This project asks: *can a model learn useful anatomical structure from unlabelled MRI scans, then need only a handful of labelled examples to segment tumours well?*

The answer, demonstrated here on the BraTS glioma dataset, is yes — an MAE-pretrained ResUNet achieves a **3.3× higher Dice score than a U-Net trained from scratch** when only 5% of labels are available.

## Pipeline Overview

```
BraTS Dataset
     │
     ▼
Data Preparation (normalization, slicing, augmentation)
     │
     ▼
MAE Pretraining (ViT encoder, unlabelled slices)
     │
     ▼
Feature Extraction (CLS token embeddings)
     │
     ▼
Pseudo-Labelling (cosine similarity, confidence ≥ 0.70, capped at 300 samples)
     │
     ▼
ResUNet Fine-Tuning (ResNet34 encoder + U-Net decoder)
     │
     ▼
Brain Tumour Segmentation (4-class voxel-wise masks)
```

1. **MAE pretraining** — a Vision Transformer (ViT) encoder learns to reconstruct randomly masked (75%) 2D axial MRI patches from unlabelled BraTS volumes, forcing it to learn rich, global anatomical representations rather than local pixel correlations.
2. **Feature extraction** — the pretrained encoder produces a CLS-token embedding per slice for both labelled and unlabelled data.
3. **Pseudo-labelling** — a cosine-similarity matrix between unlabelled and labelled feature sets flags high-confidence unlabelled slices (≥0.70 mean top-k similarity) as reliable pseudo-label candidates, capped at 300 samples to avoid overwhelming the labelled set.
4. **ResUNet fine-tuning** — the pretrained encoder is transferred into a ResNet34-encoder / U-Net-decoder segmentation model, initialized with ImageNet weights and fine-tuned on a configurable fraction of labelled data (5% or 100% in this study).

## Dataset

- **Source:** BraTS-GLI (Glioma) — a standard benchmark for brain tumour AI research, in NIfTI format.
- **Modalities:** T1-weighted (T1), contrast-enhanced T1 (T1ce), T2-weighted (T2), and FLAIR — each capturing complementary tissue characteristics (FLAIR for peritumoural edema, T1ce for enhancing tumour core).
- **Segmentation classes** (RSNA-ASNR-MICCAI convention):
  - Class 0 — Background (non-tumour)
  - Class 1 — Necrotic core (NCR)
  - Class 2 — Peritumoural edema (ED)
  - Class 3 — Enhancing tumour (ET)

### Preprocessing

- Load raw NIfTI volumes, normalize intensities to [0, 1], clip outlier percentiles
- Extract 2D axial slices for MAE pretraining, filtering for sufficient foreground content
- Medical-specific augmentation pipeline for pretraining: random gamma correction, random intensity shift, Gaussian noise, random ghosting, standard flips (simulating MRI artifacts and scanner variability)
- Lighter augmentation for fine-tuning: horizontal/vertical flips, minor intensity jitter

## Model Details

**MAE (pretraining)**
- ViT backbone; encoder processes only visible (unmasked) patches
- Learnable mask tokens + positional embeddings reconstruct masked regions via a lightweight decoder
- Training objective: MSE loss on masked patches only

**ResUNet (fine-tuning)**
- ResNet34 encoder (ImageNet-initialized) + U-Net-style decoder with skip connections
- Combined **Dice + Cross-Entropy loss** with inverse-frequency class weights to address class imbalance across tumour subregions
- **Label smoothing** to reduce overconfidence and improve generalization
- **AdamW** optimizer with differentiated learning rates (conservative for the pretrained encoder, higher for the decoder trained from scratch)
- **OneCycleLR** schedule (warmup + cosine annealing) for stable convergence
- Mixed-precision training and gradient clipping for efficiency and stability
- Best checkpoint selected by validation Dice score, not final epoch

**U-Net (baseline)**
- Classic symmetric encoder-decoder with skip connections, trained from scratch, for direct comparison against the MAE-pretrained ResUNet

## Evaluation

- **Split:** 80% training / 20% validation on the labelled dataset
- **Metrics:**
  - **Dice Similarity Coefficient** — volumetric overlap between predicted and ground-truth masks
  - **IoU (Jaccard Index)** — spatial overlap, similar to Dice
  - **95th-percentile Hausdorff Distance (HD95)** — boundary precision, robust to outliers

## Results

| Label fraction | Dice (ResUNet) | Dice (U-Net) | IoU (ResUNet) | HD95 (ResUNet, voxels) |
|---|---|---|---|---|
| 5% | **0.1279** | 0.0393 | 0.0683 | 26.16 |
| 100% | **0.2601** | 0.1258 | 0.1495 | 22.2 |

**Key findings:**
- At 5% labelled data, the MAE-pretrained ResUNet achieves a Dice score **3.3× higher** than a U-Net trained from scratch — strong evidence that self-supervised pretraining provides a useful inductive bias in data-scarce regimes.
- At 100% labelled data, ResUNet still outperforms U-Net, showing the pretrained encoder remains beneficial even under full supervision, though the performance gap narrows.
- ResUNet produces sharper, more spatially coherent tumour boundaries — particularly for the enhancing tumour (ET) and necrotic core (NCR) subregions — while U-Net tends to over-segment peritumoral edema and fragment smaller subregions.
- MAE reconstruction loss decreases steadily during pretraining (≈1.34 → ≈0.98 over 2 epochs) without fully converging, suggesting further pretraining would likely yield additional gains.

## Requirements

- Python 3.x
- PyTorch
- A Vision Transformer implementation (for the MAE encoder)
- MONAI or equivalent medical-imaging utilities (NIfTI I/O, augmentations)
- NumPy, scikit-learn (cosine similarity for pseudo-labelling)
- GPU strongly recommended (ViT + ResNet34/U-Net training)

## Limitations & Future Work

- MAE pretraining was run for only 2 epochs in this study; loss curves indicate the model had not yet converged, so longer pretraining is a likely source of further improvement.
- The pseudo-labelling confidence threshold (0.70) and sample cap (300) were fixed heuristics — adaptive thresholds could improve pseudo-label quality.
- Future work includes testing transferability across other medical imaging modalities and exploring more sophisticated fine-tuning strategies to further exploit the pretraining/fine-tuning synergy.

## References

Key related work includes MAE (He et al., 2022), Vision Transformers (Dosovitskiy et al., 2021), U-Net (Ronneberger et al., 2015), the BraTS 2021 benchmark (Baid et al., 2021), and the MONAI framework (Cardoso et al., 2022). 



