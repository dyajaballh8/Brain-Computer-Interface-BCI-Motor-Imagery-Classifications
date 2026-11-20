# Brain-Computer Interface (BCI) — EEG Motor Imagery Classification

An end-to-end Machine Learning & Deep Learning pipeline for classifying **Motor Imagery EEG signals** (`Left` vs. `Right` hand movement imagination).

---

## Folder Structure

```
EEG_Task/
  01_EEG_EDA.ipynb             Exploratory data analysis (8 cells)
  02_EEG_Preprocessing.ipynb   Signal preprocessing & QC pipeline (7 cells)
  03_EEG_Modeling.ipynb        Model training, cross-validation & evaluation (8 cells)
  run_modeling_experiment.py   Standalone Python script to run full modeling pipeline
  labels.csv                   Ground-truth labels (Left / Right) per trial
  trial_label_mapping.csv      Filename-to-label mapping table
  requirements.txt             Python dependencies
  README_01_EEG_EDA.md         Detailed documentation for EDA
  README_02_EEG_Preprocessing.md Detailed documentation for Preprocessing
  README_03_EEG_Modeling.md    Detailed documentation for Modeling & Benchmark
  data/eeg_files/              Raw EEG Excel trial files
  processed_data/              Preprocessed arrays (X_eeg.npy, y_eeg.npy, config, channels)
  models/                      Trained model artifacts, leaderboard, plots & test metrics
```

---

## Workflow Overview

1. **01 — EDA (`01_EEG_EDA.ipynb`):**
   - Natural alphanumeric alignment of trials with labels.
   - Channel layout consistency check (`C3, C4, F3, F4, P3, P4, O1, O2`).
   - Signal quality audit (NaNs, flat channels, clipping, duration).
   - Frequency analysis & verification of contralateral Event-Related Desynchronization (ERD).

2. **02 — Preprocessing (`02_EEG_Preprocessing.ipynb`):**
   - Common Average Reference (CAR) spatial filtering.
   - 50 Hz IIR Notch filter (mains noise removal).
   - 8–30 Hz zero-phase Butterworth band-pass filter (mu + beta rhythm selection).
   - Standardization: 4.0s (1024 samples) crop/pad + per-channel Z-score normalization.
   - Output: `X_eeg.npy` shape `(2400, 8, 1024)`, `y_eeg.npy` shape `(2400,)`.

3. **03 — Modeling & Evaluation (`03_EEG_Modeling.ipynb`):**
   - **Stratified Split:** 80% Train/Val (1920 trials) vs. 20% Unseen Hold-out Test (480 trials).
   - **Feature Extraction:** Sub-band Filter Bank CSP (FBCSP on mu & beta) + Welch PSD Band Power + Hemispheric Motor Asymmetry ($\Delta C3 - C4$).
   - **Benchmarking (5-Fold Stratified CV):** Logistic Regression, Linear SVM, Kernel SVM (RBF), Random Forest, LightGBM, XGBoost, and end-to-end 1D-CNN (`EEGConvNet`).
   - **Hyperparameter Optimization:** GridSearchCV tuning for top classical model.
   - **Final Test Evaluation & Neurophysiological Validation:**
     - Best Classical Model (Tuned SVM RBF): **72.71% Test Accuracy | 0.7975 ROC-AUC**
     - Best Deep Learning Model (EEGConvNet PyTorch): **92.92% Test Accuracy | 0.9794 ROC-AUC**

---

## Benchmark Results Summary

| Model | Architecture | 5-Fold CV Accuracy | Test Accuracy (Unseen) | Test ROC-AUC | Test F1-Macro |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 **EEGConvNet** | **Deep Learning (PyTorch)** | **86.41% ± 3.04%** | **92.92%** | **0.9794** | **0.9292** |
| 🥈 **Tuned SVM (RBF)** | Classical ML + FBCSP | **75.36% ± 3.37%** | **72.71%** | **0.7975** | **0.7269** |
| 🥉 **Linear SVM** | Classical ML + FBCSP | 74.27% ± 3.77% | — | — | — |
| 4 | **Random Forest** | Classical ML + FBCSP | 73.91% ± 3.00% | — | — | — |
| 5 | **LightGBM** | Classical ML + FBCSP | 73.91% ± 2.94% | — | — | — |
| 6 | **Logistic Regression** | Linear Baseline | 73.80% ± 3.97% | — | — | — |
| 7 | **XGBoost** | Classical ML + FBCSP | 73.13% ± 2.91% | — | — | — |

---

## How to Run

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute notebooks in order:**
   - Run `01_EEG_EDA.ipynb`
   - Run `02_EEG_Preprocessing.ipynb`
   - Run `03_EEG_Modeling.ipynb`
3. Or run the complete modeling pipeline via the standalone script:
   ```bash
   python run_modeling_experiment.py
   ```
