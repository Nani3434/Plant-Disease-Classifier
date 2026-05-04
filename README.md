# Plant Leaf Disease Classifier
**MSML640 Final Project — Spring 2026**
**Author:** Hari Haran Manda

---

## AI Usage Disclosure

As permitted by the course instructor, AI tools were used for the following non-core tasks only:
- Generating utility scripts (file preparation, plotting, visualization)
- README documentation formatting

The following were written and designed manually by me:
- Core training pipeline logic and all 4 configuration designs
- Model selection rationale and hyperparameter justification
- Synthetic data strategy and transformation choices
- All code comments explaining parameter decisions
- Error analysis and interpretation of results

---

## Overview

This project explores how a pretrained deep learning model (EfficientNet-B0) can be adapted to detect tomato leaf disease through fine-tuning. The system takes a single leaf photograph as input and classifies it into one of three categories — Healthy, Early Blight, or Late Blight.

The core focus is not just building a classifier, but rigorously analyzing how data augmentation and synthetic data generation affect model performance across four distinct training configurations.

---

## Project Structure

```
Plant-Disease-Classifier/
│
├── data/                        # Train / Val / Test image splits
│   ├── train/
│   │   ├── healthy/
│   │   ├── early_blight/
│   │   └── late_blight/
│   ├── val/
│   └── test/
│
├── synth_data/                  # Synthetically generated training images
│   └── train/
│       ├── healthy/
│       ├── early_blight/
│       └── late_blight/
│
├── results/                     # All plots, confusion matrices, saved models
│   ├── config1_curves.png
│   ├── config1_confusion.png
│   ├── config2_curves.png
│   ├── config2_confusion.png
│   ├── config3_curves.png
│   ├── config3_confusion.png
│   ├── config4_curves.png
│   ├── config4_confusion.png
│   ├── comparison_confusion.png
│   ├── summary_bar.png
│   ├── robustness.png
│   └── error_analysis.png
│
├── proposal/
│   └── MSML640_Proposal_HariHaranManda.docx
│
├── presentation/
│   └── slides.pptx
│
├── prepare_dataset.py           # Splits PlantVillage into train/val/test
├── generate_synth.py            # Generates synthetic training images
├── train.py                     # Trains EfficientNet-B0 under 4 configs
├── analysis.py                  # Produces all evaluation plots
└── README.md
```

---

## Problem Statement

Crop disease is one of the leading causes of agricultural yield loss worldwide. Farmers — especially in rural areas — rarely have quick access to expert diagnosis. The goal of this project is to build a lightweight, accurate image classifier that can identify tomato leaf disease from a photograph, with the potential for future deployment on a mobile device in the field.

The three classes represent progressively worsening disease states:
- **Healthy** — no disease present
- **Early Blight** — caused by *Alternaria solani*, appears as dark concentric rings
- **Late Blight** — caused by *Phytophthora infestans*, appears as irregular water-soaked patches

---

## Dataset Details

**Source:** PlantVillage Dataset — public domain, available on Kaggle

| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| Healthy | 70 | 15 | 15 | 100 |
| Early Blight | 70 | 15 | 15 | 100 |
| Late Blight | 70 | 15 | 15 | 100 |
| **Total** | **210** | **45** | **45** | **300** |

- Images are real-world RGB photographs of individual leaves under natural lighting
- Dataset is perfectly balanced — no class imbalance
- Train / Val / Test splits are strictly non-overlapping
- Images resized to 224×224 for model input

For Configs 3 and 4, an additional 1050 synthetic images were generated (350 per class) using aggressive image transformations applied to the training set.

---

## Implementation

### Backbone Model
EfficientNet-B0 pretrained on ImageNet was chosen as the backbone for the following reasons:

- **~5.3M parameters** — lightweight enough to fine-tune on a small 300-image dataset without severe overfitting
- **Compound scaling** — width, depth, and resolution are optimally balanced, making it ideal for fine-grained visual distinctions between disease stages
- **77.1% top-1 ImageNet accuracy** — outperforms ResNet-50 at a fraction of the compute cost
- **~6ms inference on CPU** — suitable for eventual mobile deployment

All backbone layers are frozen during training. Only the final classification head is trained, which is replaced with a Dropout + Linear layer for 3-class output.

### Four Training Configurations

| Config | Description |
|--------|-------------|
| 1 — Baseline | Original 300 images, no augmentation |
| 2 — Augmented | Original data + random flip, rotation, color jitter, crop, blur |
| 3 — Synthesized | Original + 1050 synthetic images, no augmentation |
| 4 — Full Pipeline | Original + 1050 synthetic images + augmentation |

### Synthetic Data Generation
Synthetic images were created by applying aggressive but realistic transformations to existing training images:
- Random flips and rotations (±45°)
- Extreme color jitter (brightness, contrast, saturation, sharpness)
- Gaussian blur (radius 0.5–3.0)
- Random Gaussian noise (sigma up to 25)
- Random rectangular erasing (simulating occlusion)

Each training image produced 5 synthetic variants, giving 350 synthetic images per class.

### Training Details

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Optimizer | Adam | Adaptive lr, stable for fine-tuning |
| Learning rate | 1e-4 | Low to avoid disrupting pretrained features |
| Scheduler | StepLR (step=5, γ=0.5) | Gradually reduces lr as training stabilizes |
| Epochs | 15 | Sufficient for convergence on small dataset |
| Batch size | 16 | Fits comfortably in CPU memory |
| Dropout | 0.3 | Mild regularization on classifier head |

---

## How to Run

### 1. Install Dependencies
```bash
python -m pip install torch torchvision pillow matplotlib scikit-learn numpy
```

### 2. Download Dataset
Download PlantVillage from Kaggle:
 https://www.kaggle.com/datasets/emmarex/plantdisease

### 3. Prepare Dataset
```bash
python prepare_dataset.py --src "path/to/PlantVillage"
```

### 4. Generate Synthetic Data
```bash
python generate_synth.py
```

### 5. Train the Model
```bash
python train.py --config 1   # Baseline
python train.py --config 2   # Augmented
python train.py --config 3   # Synthesized
python train.py --config 4   # Full Pipeline
```

### 6. Run Analysis
```bash
python analysis.py
```
All plots are saved to the `results/` folder automatically.

---

## Model Details

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B0 |
| Pretrained on | ImageNet |
| Trainable layers | Classifier head only |
| Input size | 224 × 224 × 3 |
| Output classes | 3 |
| Loss function | CrossEntropyLoss |
| Device | CPU (CUDA if available) |

---

## Results and Key Findings

### Test Accuracy Across All Configurations

| Config | Description | Test Accuracy |
|--------|-------------|---------------|
| 1 | Baseline | 73.33% |
| 2 | Augmented | 73.33% |
| **3** | **Synthesized** | **84.44% Best** |
| 4 | Full Pipeline | 75.56% |

### Key Findings

**1. Synthetic data was the most impactful improvement**
Config 3 jumped from 73.33% to 84.44% — an 11.11% gain — purely from adding synthetically transformed images. This significantly expanded the training distribution and helped the model generalize better.

**2. Augmentation alone had no effect**
Config 2 matched Config 1 exactly at 73.33%. The original dataset already contained enough natural variation that standard augmentations did not provide additional benefit.

**3. Combining synthesis and augmentation hurt performance**
Config 4 scored only 75.56% — lower than Config 3. This suggests that applying augmentation on top of already heavily transformed synthetic images introduced too much regularization, leading to underfitting.

**4. Early Blight was the easiest class to detect**
Early Blight achieved 100% recall across all configurations. Its distinct concentric dark rings are visually unique and easy for the model to identify.

**5. Healthy vs Late Blight caused the most confusion**
These two classes share similar green and brown tones, making them the hardest pair to separate — especially at early stages of infection.

**6. Motion blur caused the largest robustness drop**
Under motion blur, accuracy fell from 84.44% to 33.33%. This is the model's biggest weakness and suggests it relies heavily on sharp edge features in leaf textures.

### Robustness Testing Results (Config 3 — Best Model)

| Perturbation | Accuracy | Drop |
|--------------|----------|------|
| Clean (no perturbation) | 84.44% | — |
| Gaussian Noise | 53.33% | -31.11% |
| Motion Blur | 33.33% | -51.11% |
| Occlusion | 73.33% | -11.11% |
| Low Brightness | 73.33% | -11.11% |

The model handles occlusion and brightness shifts reasonably well but is highly sensitive to blur and noise — both of which destroy the fine texture patterns the model relies on for leaf disease detection.
