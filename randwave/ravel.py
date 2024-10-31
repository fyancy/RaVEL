
import numpy as np
# from numba import njit, prange
from .kernel_fit import fit_kernels
from .kernel_apply import apply_kernels_multi


class RandWaveTransform:
    def __init__(self, num_kernels: int=250, num_diff: int=3):
        """
        The implementation for the RandWave transformation. Refer to the paper
        "Beyond deep features: Fast random wavelet kernel convolution for weak-fault feature extraction of rotating machinery"
        by Feng et al. for more details.

        :param num_kernels: int, default 250
        :param num_diff: int, default 3
        """

        self.num_kernels = num_kernels
        self.num_diff = num_diff
        self.kernels = [None, None]
        self.mean, self.std = None, None

    def normalize(self, x):
        x = x.reshape(-1, x.shape[-1])
        X_tra_norm = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-12)
        X_tra_diff = np.diff(x, self.num_diff)
        return X_tra_norm, X_tra_diff

    def fit_kernels(self, x1: np.ndarray, x2: np.ndarray):
        k1 = fit_kernels(x1, self.num_kernels, multiwavelet=True)
        k2 = fit_kernels(x2, self.num_kernels, multiwavelet=True)
        self.kernels = [k1, k2]

    def apply_kernels(self, x1, x2, kernels):
        x1 = apply_kernels_multi(x1, kernels[0], cosine_pool=True)
        x2 = apply_kernels_multi(x2, kernels[1], cosine_pool=True)
        return np.concatenate([x1, x2], axis=1)

    def fit(self, x: np.ndarray):
        """
        Fit the kernels for the input data x
        :param x: x_train, np.ndarray, [N, L]
        :return:
        """
        x1, x2 = self.normalize(x)
        self.fit_kernels(x1, x2)
        f = self.apply_kernels(x1, x2, self.kernels)
        self.mean, self.std = f.mean(0), f.std(0) + 1e-12

        return (f - self.mean) / self.std

    def fit_transform(self, x: np.ndarray):
        """
        Fit the kernels for the input data x and transform it
        :param x: x_train, np.ndarray, [N, L]
        :return:
        """
        return self.fit(x)

    def transform(self, x: np.ndarray):
        x1, x2 = self.normalize(x)
        f = self.apply_kernels(x1, x2, self.kernels)
        f = (f - self.mean) / self.std
        return f


def rand_wave_transform(X_train, X_val, num_kernels=250, num_diff=3):
    """
    Perform the RandWave transformation on input data.

    Parameters:
    - X_train (np.ndarray): Training data of shape [N, L]
    - X_val (np.ndarray): Validation data of shape [M, L]
    - num_kernels (int): Number of kernels to use in the transformation
    - num_diff (int): Number of times to apply the difference transformation

    Returns:
    - X_tra_feat (np.ndarray): Transformed features for training data
    - X_val_feat (np.ndarray): Transformed features for validation data
    """

    # Normalize and compute difference transformation
    def normalize_and_diff(x):
        x = x.reshape(-1, x.shape[-1])
        X_tra_norm = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-12)
        X_tra_diff = np.diff(x, n=num_diff, axis=1)
        return X_tra_norm, X_tra_diff

    # Apply normalization and difference to training and validation data
    X_tra_norm, X_tra_diff = normalize_and_diff(X_train)
    X_val_norm, X_val_diff = normalize_and_diff(X_val)

    # Fit random wavelet kernels (assuming `fit_kernels` is an external function)
    k1 = fit_kernels(X_tra_norm, num_kernels, multiwavelet=True)
    k2 = fit_kernels(X_tra_diff, num_kernels, multiwavelet=True)
    kernels = [k1, k2]

    # Apply kernels to data (assuming `apply_kernels_multi` is an external function)
    def apply_kernels(x_norm, x_diff, kernels):
        x1 = apply_kernels_multi(x_norm, kernels[0], cosine_pool=True)
        x2 = apply_kernels_multi(x_diff, kernels[1], cosine_pool=True)
        return np.concatenate([x1, x2], axis=1)

    # Apply transformations to training and validation data
    X_tra_feat = apply_kernels(X_tra_norm, X_tra_diff, kernels)
    X_val_feat = apply_kernels(X_val_norm, X_val_diff, kernels)

    # Normalize transformed features
    mean, std = X_tra_feat.mean(axis=0), X_tra_feat.std(axis=0) + 1e-12
    X_tra_feat = (X_tra_feat - mean) / std
    X_val_feat = (X_val_feat - mean) / std

    return X_tra_feat, X_val_feat
