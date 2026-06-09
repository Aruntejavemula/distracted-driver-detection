# Distracted Driver Detection

> Udacity Machine Learning Engineer Nanodegree — Capstone Project
> **Best Kaggle log-loss: 1.125** (58% improvement over baseline)

[![Python](https://img.shields.io/badge/Python-3.6-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-1.1-orange.svg)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.0.4-red.svg)](https://keras.io)
[![Kaggle](https://img.shields.io/badge/Kaggle-State%20Farm-20BEFF.svg)](https://www.kaggle.com/c/state-farm-distracted-driver-detection)

---

## Overview

Distracted driving kills approximately 3,000 people in the United States every year. This project builds a deep learning pipeline to **automatically classify driver behavior** from a single dashboard camera image into one of 10 categories — using CNNs trained from scratch and VGG16 transfer learning.

This was submitted as a capstone project for the Udacity Machine Learning Engineer Nanodegree and entered into the [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection) Kaggle competition.

---

## The 10 Behavioral Classes

| Code | Behavior | Code | Behavior |
|------|----------|------|----------|
| C0 | Safe Driving | C5 | Adjusting Radio |
| C1 | Texting — Right Hand | C6 | Drinking |
| C2 | Phone Call — Right Hand | C7 | Reaching Behind |
| C3 | Texting — Left Hand | C8 | Hair & Makeup |
| C4 | Phone Call — Left Hand | C9 | Talking to Passenger |

---

## Dataset

- **Source:** [Kaggle — State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)
- **Total images:** 102,150 dashboard camera photographs (640×480, RGB)
- **Train / Val / Test:** 17,939 / 4,485 / 79,726 images
- **Note:** Multiple drivers per class — prevents identity-based overfitting

---

## Methodology

### Phase 1 — Data Preparation
- 80/20 train/validation split (`random_state=42`)
- Zero-mean normalization per channel
- Resize all images to 224×224 (VGG16 input requirement)
- Pre-compute and save VGG16 bottleneck features for all splits

### Phase 2 — Models

**CNN from Scratch (Baseline)**
```
Input (224×224×3) → Conv2D(64, 3×3, ReLU) → MaxPool
→ Conv2D(128, 3×3, ReLU) → MaxPool → Flatten
→ Dense(512, ReLU) + Dropout(0.5) → Dense(10, Softmax)
```
Optimizer: RMSProp | Loss: Categorical cross-entropy

**VGG16 Transfer Learning**
1. Load VGG16 without top layers (`include_top=False`)
2. Pre-compute bottleneck features — output shape `(n, 7, 7, 512)`
3. Top model: GlobalAvgPooling2D → Dense(256, ReLU) + Dropout → Dense(10, Softmax)
4. Train top model with frozen VGG16 weights

**Fine-Tuning**
- Unfreeze VGG16 Block 5 (last 3 convolutional layers)
- Retrain end-to-end at learning rate ≈ 1×10⁻⁵
- Xavier initialization + zero-mean normalization for top layers

### Phase 3 — Evaluation
- Metric: multi-class logarithmic loss (lower = better; random = ~2.30, perfect = 0.00)
- Generate softmax probability predictions on 79,726 test images
- Submit `submission.csv` to Kaggle

---

## Results

| Submission | Description | Private Score | Public Score |
|------------|-------------|:---:|:---:|
| CNN Scratch | Conv from scratch, Xavier init, zero-mean | 3.109 | 2.671 |
| VGG16 Sub 1 | Bottleneck + GAP + 1 Dense layer | 1.821 | 1.932 |
| VGG16 Sub 2 | Bottleneck + Flatten + 2 Dense layers | 5.052 | 5.216 |
| **VGG16 Fine-Tuned** | **Fine-tuned Block 5, Xavier + zero-mean** | **1.125** | **1.264** |

**Best result:** VGG16 fine-tuned — private log-loss **1.125** — a **58% improvement** over the CNN baseline.

---

## Key Findings

1. **Transfer learning dramatically outperforms scratch training** — VGG16 bottleneck features (fixed weights) achieved 1.932 log-loss vs. 2.671 from scratch, a 42% improvement on the same small dataset.
2. **Fine-tuning adds meaningful signal** — Unfreezing just Block 5 and retraining end-to-end pushed the score from 1.932 → 1.125.
3. **Top model architecture matters** — Global average pooling (Sub 1: 1.821) far outperformed Flatten + 2 Dense (Sub 2: 5.052).
4. **Initialization strategy matters** — Xavier init + zero-mean normalization consistently produced better convergence than default random init.

---

## Repository Contents

| File | Description |
|------|-------------|
| `distracted_driver_detector_app.ipynb` | Main notebook — full pipeline with saved outputs |
| `proposal.pdf` | Udacity project proposal |
| `report.pdf` | Final project report |
| `Kaggle_Submission_Result.png` | Screenshot of Kaggle submission scores |
| `presentation.md` | Presentation-style summary of the project |

---

## Environment

> ⚠️ Built in 2017 with TF 1.1 / Keras 2.0.4. Will **not run as-is** on modern TensorFlow without porting to the TF 2.x API.

```
Python        3.6.0  (Anaconda 4.3.1, 64-bit)
TensorFlow    1.1.0
Keras         2.0.4
```

**Data files** (not included — download from Kaggle):
- `imgs.zip` — all train/test images
- `driver_imgs_list.csv` — image → subject → class mapping
- `sample_submission.csv` — submission format

---

## References

Centers for Disease Control and Prevention. (2023). *Distracted driving*. https://www.cdc.gov/transportationsafety/distracted_driving

Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. *arXiv:1409.1556*. https://arxiv.org/abs/1409.1556

State Farm. (2016). *State Farm distracted driver detection* [Kaggle competition]. https://www.kaggle.com/c/state-farm-distracted-driver-detection

Udacity. (2017). *Machine learning engineer nanodegree*. https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t

---

*Author: Arun Teja Vemula | Udacity ML Nanodegree Capstone*
