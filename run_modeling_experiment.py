"""
EEG Modeling Pipeline Execution Script (Optimized & Robust)
Trains, evaluates, tunes, and exports ML and DL models on processed EEG data.
"""

import os
import time
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg, signal

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration & Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

PROCESSED_DIR = "EEG_Task/processed_data"
MODELS_DIR = "EEG_Task/models"
os.makedirs(MODELS_DIR, exist_ok=True)

print("="*75)
print("03 - EEG MODELING & EVALUATION BENCHMARK PIPELINE")
print("="*75)

# Load data
X = np.load(os.path.join(PROCESSED_DIR, "X_eeg.npy"))  # (2400, 8, 1024)
y = np.load(os.path.join(PROCESSED_DIR, "y_eeg.npy"))  # (2400,)
with open(os.path.join(PROCESSED_DIR, "channels.json"), "r") as f:
    CHANNELS = json.load(f)
with open(os.path.join(PROCESSED_DIR, "config.json"), "r") as f:
    CONFIG = json.load(f)

FS = CONFIG.get("fs", 256)
N_TRIALS, N_CHANNELS, N_SAMPLES = X.shape
print(f"Loaded clean EEG data: {N_TRIALS} trials, {N_CHANNELS} channels ({CHANNELS}), {N_SAMPLES} samples @ {FS}Hz")
print(f"Class distribution: Left (0) = {np.sum(y == 0)}, Right (1) = {np.sum(y == 1)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Stratified Data Split (80% Train/Val, 20% Hold-out Test)
# ─────────────────────────────────────────────────────────────────────────────
train_idx, test_idx, y_train, y_test = train_test_split(
    np.arange(N_TRIALS), y, test_size=0.20, random_state=SEED, stratify=y
)

X_train, X_test = X[train_idx], X[test_idx]
print(f"Train/Val split: {len(train_idx)} trials | Test split: {len(test_idx)} trials (Hold-out)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Deterministic Pre-Filtering & Spectral Extraction
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 1] Pre-filtering sub-bands & computing spectral power...")
t0_fe = time.time()

# Mu (8-12 Hz) & Beta (16-24 Hz) Butterworth filters
sos_mu = signal.butter(4, [8, 12], btype='bandpass', fs=FS, output='sos')
sos_beta = signal.butter(4, [16, 24], btype='bandpass', fs=FS, output='sos')

X_mu = signal.sosfiltfilt(sos_mu, X, axis=-1)
X_beta = signal.sosfiltfilt(sos_beta, X, axis=-1)

# Welch PSD features across all 8 channels
freqs, psd = signal.welch(X, fs=FS, nperseg=256, axis=-1)
mu_mask = (freqs >= 8) & (freqs <= 12)
beta_mask = (freqs >= 13) & (freqs <= 30)

mu_power = np.mean(psd[:, :, mu_mask], axis=-1)   # (2400, 8)
beta_power = np.mean(psd[:, :, beta_mask], axis=-1) # (2400, 8)

log_mu = np.log(mu_power + 1e-8)
log_beta = np.log(beta_power + 1e-8)

c3_idx, c4_idx = CHANNELS.index("C3"), CHANNELS.index("C4")
f3_idx, f4_idx = CHANNELS.index("F3"), CHANNELS.index("F4")
p3_idx, p4_idx = CHANNELS.index("P3"), CHANNELS.index("P4")

# Hemispheric Asymmetries
asym_c_mu = log_mu[:, c3_idx] - log_mu[:, c4_idx]
asym_c_beta = log_beta[:, c3_idx] - log_beta[:, c4_idx]
asym_f_mu = log_mu[:, f3_idx] - log_mu[:, f4_idx]
asym_p_mu = log_mu[:, p3_idx] - log_mu[:, p4_idx]

F_spec = np.hstack([
    log_mu,
    log_beta,
    asym_c_mu[:, None],
    asym_c_beta[:, None],
    asym_f_mu[:, None],
    asym_p_mu[:, None]
])

print(f"Pre-filtering and Spectral features extracted in {time.time() - t0_fe:.2f}s! Shape: {F_spec.shape}")

# Feature names for interpretability
FEATURE_NAMES = ["CSP_Mu_1", "CSP_Mu_2", "CSP_Beta_1", "CSP_Beta_2"]
FEATURE_NAMES += [f"PSD_Mu_{ch}" for ch in CHANNELS]
FEATURE_NAMES += [f"PSD_Beta_{ch}" for ch in CHANNELS]
FEATURE_NAMES += ["Asym_Mu_C3_C4", "Asym_Beta_C3_C4", "Asym_Mu_F3_F4", "Asym_Mu_P3_P4"]

# ─────────────────────────────────────────────────────────────────────────────
# 4. Common Spatial Pattern (CSP) Class
# ─────────────────────────────────────────────────────────────────────────────
class CSP:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.filters_ = None

    def fit(self, X_trials, y_labels):
        covs = []
        for c in [0, 1]:
            Xc = X_trials[y_labels == c]
            cov_c = np.mean([
                np.dot(tr, tr.T) / (np.trace(np.dot(tr, tr.T)) + 1e-10)
                for tr in Xc
            ], axis=0)
            covs.append(cov_c)
        cov_tot = covs[0] + covs[1]
        evals, evecs = linalg.eigh(cov_tot)
        P = np.dot(np.diag(1.0 / np.sqrt(evals[::-1] + 1e-10)), evecs[:, ::-1].T)
        S0 = np.dot(P, np.dot(covs[0], P.T))
        evals_s, B = linalg.eigh(S0)
        W = np.dot(B[:, ::-1].T, P)
        m = self.n_components // 2
        self.filters_ = np.vstack([W[:m], W[-m:]])
        return self

    def transform(self, X_trials):
        feats = []
        for tr in X_trials:
            Z = np.dot(self.filters_, tr)
            var = np.var(Z, axis=1)
            feats.append(np.log((var + 1e-10) / (np.sum(var) + 1e-10)))
        return np.array(feats)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Multi-Model 5-Fold Stratified Cross-Validation (Zero Leakage)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-"*75)
print("BENCHMARKING CLASSICAL ML MODELS (5-Fold Stratified CV)")
print("-"*75)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

models_dict = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Linear SVM": LinearSVC(random_state=SEED, max_iter=2000),
    "SVM (RBF Kernel)": SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED),
    "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=SEED, verbose=-1),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=SEED, eval_metric="logloss")
}

model_results = {name: {"acc": [], "auc": [], "f1": []} for name in models_dict}

# Pre-extract fold features so each fold's CSP is fitted once
for fold_idx, (tr_sub, val_sub) in enumerate(cv.split(train_idx, y_train)):
    tr_indices = train_idx[tr_sub]
    val_indices = train_idx[val_sub]
    y_tr_fold = y[tr_indices]
    y_val_fold = y[val_indices]

    # Fit CSP strictly on train fold
    csp_m = CSP(2).fit(X_mu[tr_indices], y_tr_fold)
    csp_b = CSP(2).fit(X_beta[tr_indices], y_tr_fold)

    f_csp_tr = np.hstack([csp_m.transform(X_mu[tr_indices]), csp_b.transform(X_beta[tr_indices])])
    f_csp_val = np.hstack([csp_m.transform(X_mu[val_indices]), csp_b.transform(X_beta[val_indices])])

    feat_tr = np.hstack([f_csp_tr, F_spec[tr_indices]])
    feat_val = np.hstack([f_csp_val, F_spec[val_indices]])

    scaler = StandardScaler()
    feat_tr_s = scaler.fit_transform(feat_tr)
    feat_val_s = scaler.transform(feat_val)

    for name, clf in models_dict.items():
        clf.fit(feat_tr_s, y_tr_fold)
        preds = clf.predict(feat_val_s)
        acc = accuracy_score(y_val_fold, preds)
        f1 = f1_score(y_val_fold, preds, average="macro")

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(feat_val_s)[:, 1]
            auc = roc_auc_score(y_val_fold, probs)
        elif hasattr(clf, "decision_function"):
            scores = clf.decision_function(feat_val_s)
            auc = roc_auc_score(y_val_fold, scores)
        else:
            auc = acc

        model_results[name]["acc"].append(acc)
        model_results[name]["auc"].append(auc)
        model_results[name]["f1"].append(f1)

leaderboard = []
for name in models_dict:
    mean_acc = np.mean(model_results[name]["acc"])
    std_acc = np.std(model_results[name]["acc"])
    mean_auc = np.mean(model_results[name]["auc"])
    mean_f1 = np.mean(model_results[name]["f1"])

    leaderboard.append({
        "Model": name,
        "Type": "Classical ML",
        "CV Accuracy": f"{mean_acc:.4f} ± {std_acc:.4f}",
        "CV ROC-AUC": f"{mean_auc:.4f}",
        "CV F1-Macro": f"{mean_f1:.4f}",
        "_mean_acc": mean_acc,
        "_std_acc": std_acc
    })
    print(f"[{name:<20}] CV Acc: {mean_acc:.4f} ± {std_acc:.4f} | ROC-AUC: {mean_auc:.4f} | F1: {mean_f1:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Deep Learning Benchmark (EEGConvNet / PyTorch)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-"*75)
print("BENCHMARKING DEEP LEARNING (EEGConvNet in PyTorch)")
print("-"*75)

class EEGConvNet(nn.Module):
    """1D ConvNet specifically optimized for EEG Motor Imagery oscillations."""
    def __init__(self, n_channels=8, n_classes=2):
        super().__init__()
        # Temporal Conv (8-30 Hz wave patterns)
        self.conv1 = nn.Conv1d(n_channels, 16, kernel_size=32, padding=16)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(4)

        # Spatial/Channel Conv
        self.conv2 = nn.Conv1d(16, 32, kernel_size=16, padding=8)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=8, padding=4)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.AdaptiveAvgPool1d(4)

        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(64 * 4, n_classes)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.pool1(self.act(self.bn1(self.conv1(x))))
        x = self.pool2(self.act(self.bn2(self.conv2(x))))
        x = self.pool3(self.act(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = self.drop(x)
        return self.fc(x)

dl_accs, dl_aucs, dl_f1s = [], [], []
for fold_idx, (tr_sub, val_sub) in enumerate(cv.split(train_idx, y_train)):
    tr_indices = train_idx[tr_sub]
    val_indices = train_idx[val_sub]

    tr_ds = TensorDataset(torch.tensor(X[tr_indices], dtype=torch.float32), torch.tensor(y[tr_indices], dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X[val_indices], dtype=torch.float32), torch.tensor(y[val_indices], dtype=torch.long))

    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    dl_net = EEGConvNet(n_channels=N_CHANNELS)
    optimizer = optim.Adam(dl_net.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        dl_net.train()
        for bx, by in tr_loader:
            optimizer.zero_grad()
            out = dl_net(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    dl_net.eval()
    v_preds, v_probs = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            logits = dl_net(bx)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            v_preds.extend(logits.argmax(dim=-1).numpy())
            v_probs.extend(probs.numpy())

    dl_accs.append(accuracy_score(y[val_indices], v_preds))
    dl_aucs.append(roc_auc_score(y[val_indices], v_probs))
    dl_f1s.append(f1_score(y[val_indices], v_preds, average="macro"))

mean_dl_acc, std_dl_acc = np.mean(dl_accs), np.std(dl_accs)
mean_dl_auc, mean_dl_f1 = np.mean(dl_aucs), np.mean(dl_f1s)

leaderboard.append({
    "Model": "EEGConvNet (PyTorch)",
    "Type": "Deep Learning",
    "CV Accuracy": f"{mean_dl_acc:.4f} ± {std_dl_acc:.4f}",
    "CV ROC-AUC": f"{mean_dl_auc:.4f}",
    "CV F1-Macro": f"{mean_dl_f1:.4f}",
    "_mean_acc": mean_dl_acc,
    "_std_acc": std_dl_acc
})
print(f"[EEGConvNet (PyTorch) ] CV Acc: {mean_dl_acc:.4f} ± {std_dl_acc:.4f} | ROC-AUC: {mean_dl_auc:.4f} | F1: {mean_dl_f1:.4f}")

# Leaderboard Table
df_leaderboard = pd.DataFrame(leaderboard).sort_values(by="_mean_acc", ascending=False).reset_index(drop=True)
df_display = df_leaderboard.drop(columns=["_mean_acc", "_std_acc"])
df_display.to_csv(os.path.join(MODELS_DIR, "leaderboard.csv"), index=False)

print("\n" + "="*75)
print("FINAL BENCHMARK LEADERBOARD (Ranked by CV Accuracy):")
print("="*75)
print(df_display.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 7. Hyperparameter Tuning of the Top Classical Model (SVM RBF)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-"*75)
print("HYPERPARAMETER TUNING: OPTIMIZING TOP CLASSICAL MODEL (SVM RBF)")
print("-"*75)

# Full train features
full_csp_m = CSP(2).fit(X_mu[train_idx], y_train)
full_csp_b = CSP(2).fit(X_beta[train_idx], y_train)

F_tr_csp = np.hstack([full_csp_m.transform(X_mu[train_idx]), full_csp_b.transform(X_beta[train_idx])])
F_te_csp = np.hstack([full_csp_m.transform(X_mu[test_idx]), full_csp_b.transform(X_beta[test_idx])])

F_train = np.hstack([F_tr_csp, F_spec[train_idx]])
F_test = np.hstack([F_te_csp, F_spec[test_idx]])

scaler_full = StandardScaler()
F_train_s = scaler_full.fit_transform(F_train)
F_test_s = scaler_full.transform(F_test)

param_grid = {
    "C": [0.5, 1.0, 2.0, 5.0],
    "gamma": ["scale", 0.02, 0.05, 0.1]
}

grid_svm = GridSearchCV(
    SVC(kernel="rbf", probability=True, random_state=SEED),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=1
)
grid_svm.fit(F_train_s, y_train)
best_svm = grid_svm.best_estimator_

print(f"Optimal Parameters: {grid_svm.best_params_}")
print(f"Tuned SVM CV Accuracy: {grid_svm.best_score_:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Final Evaluation on Hold-Out Test Set (Unseen 480 Trials)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*75)
print("FINAL EVALUATION ON UNSEEN HOLD-OUT TEST SET (480 Trials)")
print("="*75)

# A. Classical Tuned SVM Test Evaluation
svm_preds = best_svm.predict(F_test_s)
svm_probs = best_svm.predict_proba(F_test_s)[:, 1]

svm_acc = accuracy_score(y_test, svm_preds)
svm_bal_acc = balanced_accuracy_score(y_test, svm_preds)
svm_f1 = f1_score(y_test, svm_preds, average="macro")
svm_auc = roc_auc_score(y_test, svm_probs)

print("\n--- [1] BEST CLASSICAL MODEL: Tuned SVM (RBF Kernel) ---")
print(f"Test Accuracy:         {svm_acc*100:.2f}%")
print(f"Balanced Accuracy:     {svm_bal_acc*100:.2f}%")
print(f"F1-Score (Macro):      {svm_f1:.4f}")
print(f"ROC-AUC:               {svm_auc:.4f}")
print("\nClassification Report (SVM):\n", classification_report(y_test, svm_preds, target_names=["Left", "Right"]))

# B. Deep Learning EEGConvNet Full Training & Test Evaluation
print("\n--- [2] BEST DEEP LEARNING MODEL: EEGConvNet (PyTorch) ---")
final_dl = EEGConvNet(n_channels=N_CHANNELS)
optimizer = optim.Adam(final_dl.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

train_ds_full = TensorDataset(torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_ds_full = TensorDataset(torch.tensor(X[test_idx], dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

tr_loader_full = DataLoader(train_ds_full, batch_size=64, shuffle=True)
te_loader_full = DataLoader(test_ds_full, batch_size=64, shuffle=False)

for epoch in range(12):
    final_dl.train()
    for bx, by in tr_loader_full:
        optimizer.zero_grad()
        out = final_dl(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()

final_dl.eval()
dl_preds, dl_probs = [], []
with torch.no_grad():
    for bx, by in te_loader_full:
        logits = final_dl(bx)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)
        dl_preds.extend(preds.numpy())
        dl_probs.extend(probs.numpy())

dl_preds = np.array(dl_preds)
dl_probs = np.array(dl_probs)

dl_acc = accuracy_score(y_test, dl_preds)
dl_bal_acc = balanced_accuracy_score(y_test, dl_preds)
dl_f1 = f1_score(y_test, dl_preds, average="macro")
dl_auc = roc_auc_score(y_test, dl_probs)

print(f"Test Accuracy:         {dl_acc*100:.2f}%")
print(f"Balanced Accuracy:     {dl_bal_acc*100:.2f}%")
print(f"F1-Score (Macro):      {dl_f1:.4f}")
print(f"ROC-AUC:               {dl_auc:.4f}")
print("\nClassification Report (EEGConvNet):\n", classification_report(y_test, dl_preds, target_names=["Left", "Right"]))

# Save metrics summary
metrics_summary = {
    "test_set_size": len(y_test),
    "best_classical_model": {
        "name": "Tuned SVM (RBF Kernel)",
        "params": grid_svm.best_params_,
        "test_accuracy": float(svm_acc),
        "test_balanced_accuracy": float(svm_bal_acc),
        "test_f1_macro": float(svm_f1),
        "test_roc_auc": float(svm_auc)
    },
    "best_deep_model": {
        "name": "EEGConvNet (PyTorch)",
        "test_accuracy": float(dl_acc),
        "test_balanced_accuracy": float(dl_bal_acc),
        "test_f1_macro": float(dl_f1),
        "test_roc_auc": float(dl_auc)
    }
}
with open(os.path.join(MODELS_DIR, "test_metrics.json"), "w") as f:
    json.dump(metrics_summary, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Visualizations & Interpretability
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-"*75)
print("GENERATING VISUALIZATIONS & ARTIFACTS")
print("-"*75)

# A. Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cm_svm = confusion_matrix(y_test, svm_preds)
cm_dl = confusion_matrix(y_test, dl_preds)

sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Left", "Right"], yticklabels=["Left", "Right"], ax=axes[0])
axes[0].set_title(f"Tuned SVM (RBF)\nTest Acc: {svm_acc*100:.2f}% | AUC: {svm_auc:.3f}", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

sns.heatmap(cm_dl, annot=True, fmt="d", cmap="Greens", cbar=False,
            xticklabels=["Left", "Right"], yticklabels=["Left", "Right"], ax=axes[1])
axes[1].set_title(f"EEGConvNet (PyTorch)\nTest Acc: {dl_acc*100:.2f}% | AUC: {dl_auc:.3f}", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")

plt.tight_layout()
cm_path = os.path.join(MODELS_DIR, "confusion_matrices.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"Saved: {cm_path}")

# B. ROC Curves Comparison
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_probs)
fpr_dl, tpr_dl, _ = roc_curve(y_test, dl_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, label=f"Tuned SVM RBF (AUC = {svm_auc:.3f})", color="#1f77b4", lw=2.5)
plt.plot(fpr_dl, tpr_dl, label=f"EEGConvNet PyTorch (AUC = {dl_auc:.3f})", color="#2ca02c", lw=2.5)
plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
plt.title("ROC Curves Comparison on Unseen Test Set (480 Trials)", fontsize=13, fontweight="bold")
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
roc_path = os.path.join(MODELS_DIR, "roc_curves.png")
plt.savefig(roc_path, dpi=300)
plt.close()
print(f"Saved: {roc_path}")

# C. Feature Importance & Physiological Verification
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED)
rf_model.fit(F_train_s, y_train)

feat_imp = pd.Series(rf_model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_imp.head(10).plot(kind="barh", color="#3470a3").invert_yaxis()
plt.title("Top 10 Most Important Features (Physiological Verification)", fontsize=12, fontweight="bold")
plt.xlabel("Feature Importance Score")
plt.tight_layout()
fi_path = os.path.join(MODELS_DIR, "feature_importance.png")
plt.savefig(fi_path, dpi=300)
plt.close()
print(f"Saved: {fi_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Export Models
# ─────────────────────────────────────────────────────────────────────────────
classical_bundle = {
    "csp_mu": full_csp_m,
    "csp_beta": full_csp_b,
    "scaler": scaler_full,
    "model": best_svm,
    "channels": CHANNELS,
    "feature_names": FEATURE_NAMES
}
joblib.dump(classical_bundle, os.path.join(MODELS_DIR, "best_classical_pipeline.joblib"))
torch.save(final_dl.state_dict(), os.path.join(MODELS_DIR, "best_deep_model.pt"))

print("\n" + "="*75)
print("SUCCESS: ALL MODELING TASKS COMPLETED!")
print(f"Saved best classical pipeline to: {os.path.join(MODELS_DIR, 'best_classical_pipeline.joblib')}")
print(f"Saved deep learning model weights to: {os.path.join(MODELS_DIR, 'best_deep_model.pt')}")
print(f"Saved leaderboard to: {os.path.join(MODELS_DIR, 'leaderboard.csv')}")
print(f"Saved metrics to: {os.path.join(MODELS_DIR, 'test_metrics.json')}")
print("="*75)
