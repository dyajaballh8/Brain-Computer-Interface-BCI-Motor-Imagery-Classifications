"""Feature extraction for LEFT/RIGHT motor imagery EEG.

Shared by the feature notebook, the modeling notebook and the Streamlit app,
so training and inference always use the exact same code.

Input of every transformer: X with shape (n_trials, n_channels, n_samples),
i.e. the epochs saved by main_pipeline.ipynb (outputs/epochs_CAR.npy).
"""
import numpy as np
from scipy import linalg, signal
from sklearn.base import BaseEstimator, TransformerMixin

FS = 250
CHANNELS = ["FZ", "C3", "CZ", "C4"]
BANDS = {"mu": (8, 12), "beta": (13, 30)}
FB_BANDS = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]


def bandpass(X, lo, hi, fs=FS, order=4):
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, X, axis=-1)


class CSP(BaseEstimator, TransformerMixin):
    """Common Spatial Patterns -> log-variance features.

    Supervised: fit it on training trials only (inside each CV fold).
    """

    def __init__(self, n_pairs=1, reg=1e-3):
        self.n_pairs = n_pairs
        self.reg = reg

    def _class_cov(self, X):
        covs = []
        for trial in X:
            c = np.cov(trial)
            covs.append(c / np.trace(c))
        c = np.mean(covs, axis=0)
        return c + self.reg * np.trace(c) / len(c) * np.eye(len(c))

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("CSP needs exactly 2 classes")
        ca, cb = (self._class_cov(X[y == k]) for k in self.classes_)
        evals, evecs = linalg.eigh(ca, ca + cb)
        order = np.argsort(evals)
        pick = np.r_[order[: self.n_pairs], order[-self.n_pairs:]]
        self.filters_ = evecs[:, pick].T                # (2*n_pairs, n_channels)
        self.patterns_ = linalg.pinv(self.filters_).T   # for plotting
        self.eigenvalues_ = evals[pick]
        return self

    def transform(self, X):
        Z = np.einsum("fc,ncs->nfs", self.filters_, X)
        var = Z.var(axis=-1)
        return np.log(var / var.sum(axis=1, keepdims=True))


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Band-pass the epochs into several sub-bands and run one CSP per band."""

    def __init__(self, bands=tuple(FB_BANDS), n_pairs=1, reg=1e-3, fs=FS):
        self.bands = bands
        self.n_pairs = n_pairs
        self.reg = reg
        self.fs = fs

    def fit(self, X, y):
        self.csps_ = [CSP(self.n_pairs, self.reg).fit(bandpass(X, lo, hi, self.fs), y)
                      for lo, hi in self.bands]
        return self

    def transform(self, X):
        return np.hstack([csp.transform(bandpass(X, lo, hi, self.fs))
                          for csp, (lo, hi) in zip(self.csps_, self.bands)])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"CSP_{lo}-{hi}Hz_{k + 1}" for lo, hi in self.bands
                         for k in range(2 * self.n_pairs)])


class BandPower(BaseEstimator, TransformerMixin):
    """Per-trial spectral features (no labels used -> safe to compute on all trials).

    - log band power per channel and band (Welch PSD)
    - relative band power (band / 4-40 Hz total) per channel and band
    - hemispheric asymmetry log(C3) - log(C4) per band
    """

    def __init__(self, bands=None, channels=tuple(CHANNELS), fs=FS):
        self.bands = bands
        self.channels = channels
        self.fs = fs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        bands = self.bands or BANDS
        f, P = signal.welch(X, fs=self.fs, nperseg=2 * self.fs, axis=-1)
        total = P[..., (f >= 4) & (f <= 40)].sum(-1)
        ch = list(self.channels)
        c3, c4 = ch.index("C3"), ch.index("C4")
        feats = []
        for lo, hi in bands.values():
            band = P[..., (f >= lo) & (f <= hi)].sum(-1)
            feats += [np.log(band), band / total, np.log(band[:, [c3]]) - np.log(band[:, [c4]])]
        return np.hstack(feats)

    def get_feature_names_out(self, input_features=None):
        names = []
        for b in (self.bands or BANDS):
            names += [f"logBP_{b}_{c}" for c in self.channels]
            names += [f"relBP_{b}_{c}" for c in self.channels]
            names += [f"Asym_{b}_C3-C4"]
        return np.array(names)


class ChannelStandardizer(BaseEstimator, TransformerMixin):
    """Per-channel z-score with statistics from the training trials (for deep models)."""

    def fit(self, X, y=None):
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-8
        return self

    def transform(self, X):
        return ((X - self.mean_) / self.std_).astype(np.float32)


def make_feature_union(n_pairs=1):
    """FBCSP + band power, ready to put in front of any sklearn classifier."""
    from sklearn.pipeline import FeatureUnion
    return FeatureUnion([("fbcsp", FilterBankCSP(n_pairs=n_pairs)), ("bandpower", BandPower())])
