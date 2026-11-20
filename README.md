# Brain-Computer Interface (BCI) — EEG Motor Imagery Classification (LEFT vs. RIGHT)

An end-to-end Machine Learning & Deep Learning pipeline for classifying **Motor Imagery EEG signals** (`Left` vs. `Right` hand movement imagination).

---

## Repository Structure & Pipelines

This repository provides both a **clean modular 3-stage pipeline** and comprehensive benchmark notebooks:

```
EEG_Task/
├── 01_EEG_EDA.ipynb             # Exploratory Data Analysis & signal quality audit (8 cells)
├── 02_EEG_Preprocessing.ipynb   # Signal preprocessing & artifact rejection (7 cells)
├── 03_EEG_Modeling.ipynb        # Model benchmarking, CV, tuning & evaluation (8 cells)
├── BCI_MI_Pipeline.ipynb        # Single-file comprehensive end-to-end pipeline
├── main_pipeline.ipynb          # Alternative workflow pipeline
├── run_modeling_experiment.py   # Standalone Python script for full modeling execution
├── labels.csv                   # Ground-truth trial labels (Left / Right)
├── trial_label_mapping.csv      # File-to-label mapping table
├── requirements.txt             # Python dependencies
├── README_01_EEG_EDA.md         # Detailed documentation for Notebook 01
├── README_02_EEG_Preprocessing.md # Detailed documentation for Notebook 02
├── README_03_EEG_Modeling.md    # Detailed documentation for Notebook 03
├── data/eeg_files/              # Raw EEG trial recordings
├── processed_data/              # Preprocessed arrays (X_eeg.npy, y_eeg.npy, config, channels)
└── models/                      # Serialized models, benchmark leaderboard & evaluation plots
```

---

## 3-Stage Modular Workflow

### 1. Exploratory Data Analysis (`01_EEG_EDA.ipynb`)
- Natural alphanumeric matching of trial files to labels.
- Layout consistency across channels (`C3, C4, F3, F4, P3, P4, O1, O2`).
- Audit for NaNs, flat channels, clipping, duration, and sampling rate consistency.
- Frequency spectrum and Event-Related Desynchronization (ERD) inspection on motor channels ($C3 / C4$).

### 2. Preprocessing Pipeline (`02_EEG_Preprocessing.ipynb`)
- **Common Average Reference (CAR):** Spatial re-referencing to reduce common-mode noise.
- **50 Hz Notch Filter:** Zero-phase IIR notch filter to eliminate electrical line interference.
- **8–30 Hz Band-Pass Filter:** 4th-order zero-phase Butterworth filter targeting $\mu$ (8–12 Hz) and $\beta$ (16–24 Hz) motor rhythms.
- **Standardization:** Unified 4.0-second epoching (1024 samples @ 256 Hz) with per-channel Z-score normalization.
- **Export:** Saved clean arrays `X_eeg.npy` (`2400 × 8 × 1024`) and `y_eeg.npy` (`2400`).

### 3. Modeling & Evaluation (`03_EEG_Modeling.ipynb`)
- **Strict Data Splitting:** 80% Train/Val (1920 trials) and 20% Unseen Hold-out Test (480 trials).
- **Feature Engineering:** Sub-band Filter Bank CSP (FBCSP on $\mu$ and $\beta$) + Welch PSD Band Power + Contralateral Hemispheric Asymmetry ($\Delta C3 - C4$).
- **Multi-Model Benchmark (5-Fold Stratified CV):** Logistic Regression, Linear SVM, Kernel SVM (RBF), Random Forest, LightGBM, XGBoost, and an end-to-end Deep Learning 1D-CNN (`EEGConvNet`).
- **Hyperparameter Optimization:** GridSearchCV tuning on the top classical model.
- **Neurophysiological Validation:** Verified that dominant decisions rely on contralateral motor cortex desynchronization (`Asym_Mu_C3_C4`, `CSP_Mu_1`, `PSD_Mu_C3/C4`).

---

## Model Benchmark Results

### 5-Fold Stratified Cross-Validation Leaderboard

| Model | Architecture / Type | CV Accuracy | CV ROC-AUC | CV F1-Macro |
|:---:|:---:|:---:|:---:|:---:|
| 🥇 **EEGConvNet** | **Deep Learning (PyTorch)** | **86.41% ± 3.04%** | **0.9635** | **0.8625** |
| 🥈 **Tuned SVM (RBF)** | Classical ML + FBCSP | **75.36% ± 3.37%** | **0.8152** | **0.7437** |
| 🥉 **Linear SVM** | Classical ML + FBCSP | 74.27% ± 3.77% | 0.8122 | 0.7426 |
| 4 | **Random Forest** | Classical ML + FBCSP | 73.91% ± 3.00% | 0.8163 | 0.7390 |
| 5 | **LightGBM** | Classical ML + FBCSP | 73.91% ± 2.94% | 0.8172 | 0.7389 |
| 6 | **Logistic Regression** | Linear Baseline | 73.80% ± 3.97% | 0.8118 | 0.7379 |
| 7 | **XGBoost** | Classical ML + FBCSP | 73.13% ± 2.91% | 0.8155 | 0.7312 |

### Final Evaluation on Unseen Test Set (480 Trials)

| Metric | Tuned SVM (RBF) | EEGConvNet (PyTorch) |
|---|:---:|:---:|
| **Test Accuracy** | **72.71%** | **92.92%** |
| **Balanced Accuracy** | 72.75% | **92.94%** |
| **F1-Score (Macro)** | 0.7269 | **0.9292** |
| **ROC-AUC** | 0.7975 | **0.9794** |

---

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the 3 modular notebooks:**
   - `01_EEG_EDA.ipynb`
   - `02_EEG_Preprocessing.ipynb`
   - `03_EEG_Modeling.ipynb`

3. **Or run the end-to-end Python script:**
   ```bash
   python run_modeling_experiment.py
   ```
