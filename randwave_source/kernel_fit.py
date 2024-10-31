import numpy as np
from .wavelet_fit import (fit_morlet_kernels, fit_laplace_kernels,
                          fit_hermitian_kernels, fit_harmonic_kernels)


def fit_kernels(X, num_kernels, multiscale=True, multiwavelet=False, quantile_bias=True):
    if multiscale:
        candidate_lengths = np.arange(11, 33, 2, dtype=np.int32)
    else:
        # candidate_lengths = np.arange(7, 13, dtype=np.int32)  # 7~11
        candidate_lengths = np.array([255], dtype=np.int32)  # 7~11

    if multiwavelet:
        num_kernels_per_type = num_kernels // 4
        remainder = num_kernels - num_kernels_per_type * 3

        w1, l1, b1, d1, p1 = fit_morlet_kernels(X, num_kernels_per_type, candidate_lengths, quantile_bias)
        w2, l2, b2, d2, p2 = fit_laplace_kernels(X, num_kernels_per_type, candidate_lengths, quantile_bias)
        w3, l3, b3, d3, p3 = fit_hermitian_kernels(X, num_kernels_per_type, candidate_lengths, quantile_bias)
        w4, l4, b4, d4, p4 = fit_harmonic_kernels(X, remainder, candidate_lengths, quantile_bias)

        weights, lengths, biases, dilations, paddings = np.concatenate([w1, w2, w3, w4]), \
                                                        np.concatenate([l1, l2, l3, l4]), \
                                                        np.concatenate([b1, b2, b3, b4]), \
                                                        np.concatenate([d1, d2, d3, d4]), \
                                                        np.concatenate([p1, p2, p3, p4])
    else:
        weights, lengths, biases, dilations, paddings = \
            fit_laplace_kernels(X, num_kernels, candidate_lengths, quantile_bias)
        # weights, lengths, biases, dilations, paddings = \
        #     fit_morlet_kernels(X, num_kernels, candidate_lengths, quantile_bias)

    return weights, lengths, biases, dilations, paddings
