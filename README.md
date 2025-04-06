# 🎯 SHL Hiring Assessment - Audio Regression Challenge

This repository contains my solution to the **SHL Hiring Assessment** machine learning challenge. The goal was to **predict a continuous score (label)** for given audio files, which likely reflect some human-rated property (like clarity, emotion, etc.).

## 🧠 Problem Statement

We were provided with:
- `train.csv` containing `filename` and its corresponding `label`
- `test.csv` with `filename` only
- Audio files for both train and test
- A `sample_submission.csv` format

The task was to **extract features from the audio files** and build a model to predict the label for the test set, achieving the best possible score on Kaggle's private leaderboard.

---

## 🏗️ Solution Approach

1. **Feature Extraction** using `librosa`:
   - RMS Energy
   - Zero-Crossing Rate
   - Tempo
   - 13 MFCCs (mean & std)
   - Chroma Features
   - Spectral Contrast
   - Tonnetz Features

2. **Preprocessing**:
   - Standardized using `StandardScaler`
   - Handled missing values

3. **Model**:
   - Used `LightGBM` (Gradient Boosted Decision Trees)
   - 5-fold Cross-validation with early stopping
   - Averaged predictions from all folds

4. **Evaluation**:
   - Locally evaluated using **R² Score**
   - Achieved **~0.58 R²** on Kaggle's **private leaderboard**

---

## ⚙️ Files in this Repository

| File | Description |
|------|-------------|
| `train.csv` | Training data with filenames and labels |
| `test.csv` | Test data with filenames only |
| `sample_submission.csv` | Submission format |
| `submission2.csv` | Final predictions with boosted model |
| `boosting_model.py` | Core script for feature extraction, model training, prediction |
| `README.md` | This file 😊 |

---

## 💡 Learnings & Challenges

This project was a rollercoaster. I learned a ton about:
- Extracting rich audio features using `librosa`
- Fine-tuning models with `LightGBM`
- The pain of debugging audio errors 😤 (especially due to corrupt/missing files)
- Handling overfitting and dealing with low local vs public leaderboard scores

After trying multiple models including CatBoost and Ridge, **LightGBM + extracted features** gave the best generalization.

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Local R² | ~0.61 |
| Kaggle R² (Private LB) | **0.58** ✅ |

---

## 🚀 How to Run

```bash
pip install -r requirements.txt  # Make sure you have librosa, pandas, lightgbm, sklearn, etc.

python boosting_model.py         # Outputs submission2.csv

