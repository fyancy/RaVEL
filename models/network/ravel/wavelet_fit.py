import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from numba import njit, prange

from wavelets_base import Morlet, LaplaceWavelet, \
    HermitianRe, HermitianIm, HarmonicRe, HarmonicIm


# ---------------------------------
# low-discrepancy sequence to assign quantiles to kernel/dilation combinations
def _quantiles(n):
    return np.array([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32)


# @njit(fastmath=True)
@njit("float32[:](float32[:],float32[:],int32,int32,int32)",
      fastmath=True, parallel=False, cache=True)
def _apply_kernel(X, weights, length, dilation, padding):
    input_length = len(X)
    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)

    end = (input_length + padding) - ((length - 1) * dilation)
    out = np.zeros(output_length, dtype=np.float32)

    for i in prange(-padding, end):  # i为kernel起点
        _sum = 0.
        index = i

        for j in prange(length):
            if -1 < index < input_length:
                _sum = _sum + weights[j] * X[index]  # / length NOT RECOMMEND
            index = index + dilation

        out[i + padding] = _sum
    # print(f"OUT min {min(out)}, max {max(out)}")
    return out


# --------------------------------

# BIAS_RANGE = [0, np.pi * 2]
BIAS_RANGE = [-1, 1]


def fit_morlet_kernels(X, num_kernels, candidate_lengths, quantile_bias=True):
    num_examples, input_length = X.shape

    # assert max(candidate_lengths) <= MAX_KERNEL_LEN  # restrict the length of dilations
    lengths = np.random.choice(candidate_lengths, num_kernels)

    weights = np.zeros(lengths.sum(), dtype=np.float32)
    biases = np.zeros(num_kernels, dtype=np.float32)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    fact_a = np.random.uniform(1, 10, num_kernels)
    fact_b = np.random.uniform(-5, 5, num_kernels)

    quantiles = _quantiles(num_kernels)

    a1 = 0
    for i in range(num_kernels):
        _length = lengths[i]

        _timestamps = np.linspace(0, _length // 2, num=_length)  # use this, 82.18%, sq, snr=-10
        _timestamps = (_timestamps + fact_b[i]) / fact_a[i]
        _weights = Morlet(_timestamps) * np.sqrt(1 / fact_a[i])

        _weights = (_weights - _weights.mean()) / (_weights.std() + 1e-12)
        b1 = a1 + _length
        weights[a1:b1] = _weights

        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        if quantile_bias:
            _C = _apply_kernel(X[np.random.randint(num_examples)], weights[a1:b1], _length, dilation, padding)
            biases[i] = np.quantile(_C, quantiles[i])
        else:
            biases[i] = np.random.uniform(BIAS_RANGE[0], BIAS_RANGE[1])

        a1 = b1

    return weights, lengths, biases, dilations, paddings


def fit_laplace_kernels(X, num_kernels, candidate_lengths, quantile_bias=True):
    num_examples, input_length = X.shape

    # assert max(candidate_lengths) <= MAX_KERNEL_LEN  # restrict the length of dilations
    lengths = np.random.choice(candidate_lengths, num_kernels)

    weights = np.zeros(lengths.sum(), dtype=np.float32)
    biases = np.zeros(num_kernels, dtype=np.float32)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    fact_a = np.random.uniform(0.1, 10, num_kernels)  # for Laplace; single-side Morlet/Hermit/Harmonic
    fact_b = np.random.uniform(0, 5, num_kernels)  # todo: modify

    quantiles = _quantiles(num_kernels)

    a1 = 0
    for i in range(num_kernels):
        _length = lengths[i]

        _timestamps = np.linspace(0, 1, num=_length)
        _timestamps = (_timestamps - fact_b[i]) / fact_a[i]   # + or - is OK
        _weights = LaplaceWavelet(_timestamps) * np.sqrt(1 / fact_a[i])

        _weights = (_weights - _weights.mean()) / (_weights.std() + 1e-12)
        b1 = a1 + _length
        weights[a1:b1] = _weights

        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        if quantile_bias:
            _C = _apply_kernel(X[np.random.randint(num_examples)], weights[a1:b1], _length, dilation, padding)
            biases[i] = np.quantile(_C, quantiles[i])
        else:
            biases[i] = np.random.uniform(BIAS_RANGE[0], BIAS_RANGE[1])

        a1 = b1

    return weights, lengths, biases, dilations, paddings


def fit_hermitian_kernels(X, num_kernels, candidate_lengths, quantile_bias=True):
    num_examples, input_length = X.shape

    # assert max(candidate_lengths) <= MAX_KERNEL_LEN  # restrict the length of dilations
    lengths = np.random.choice(candidate_lengths, num_kernels)

    weights = np.zeros(lengths.sum(), dtype=np.float32)
    biases = np.zeros(num_kernels, dtype=np.float32)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    fact_a = np.random.uniform(5, 10, num_kernels)  # for Laplace; single-side Morlet/Hermit/Harmonic
    fact_b = np.random.uniform(-5, 5, num_kernels)

    quantiles = _quantiles(num_kernels)

    a1 = 0
    for i in range(num_kernels):
        _length = lengths[i]

        _timestamps = np.linspace(0, _length - 1, num=_length)  # use this, better than double-side for Hermit
        _timestamps = (_timestamps - fact_b[i]) / fact_a[i]
        _weights = HermitianIm(_timestamps) * np.sqrt(1 / fact_a[i])

        _weights = (_weights - _weights.mean()) / (_weights.std() + 1e-12)
        b1 = a1 + _length
        weights[a1:b1] = _weights

        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        if quantile_bias:
            _C = _apply_kernel(X[np.random.randint(num_examples)], weights[a1:b1], _length, dilation, padding)
            biases[i] = np.quantile(_C, quantiles[i])
        else:
            biases[i] = np.random.uniform(BIAS_RANGE[0], BIAS_RANGE[1])

        a1 = b1

    return weights, lengths, biases, dilations, paddings


def fit_harmonic_kernels(X, num_kernels, candidate_lengths, quantile_bias=True):
    num_examples, input_length = X.shape

    # assert max(candidate_lengths) <= MAX_KERNEL_LEN  # restrict the length of dilations
    lengths = np.random.choice(candidate_lengths, num_kernels)

    weights = np.zeros(lengths.sum(), dtype=np.float32)
    biases = np.zeros(num_kernels, dtype=np.float32)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    fact_a = np.random.uniform(1, 10, num_kernels)
    fact_b = np.random.uniform(-5, 5, num_kernels)  # for Harmonic single-side

    quantiles = _quantiles(num_kernels)

    a1 = 0
    for i in range(num_kernels):
        _length = lengths[i]

        _timestamps = np.linspace(0, _length // 2, num=_length)  # use this, better
        _timestamps = (_timestamps + fact_b[i]) / fact_a[i]
        _weights = HarmonicRe(_timestamps) * np.sqrt(1 / fact_a[i])  # better

        _weights = (_weights - _weights.mean()) / (_weights.std() + 1e-12)
        b1 = a1 + _length
        weights[a1:b1] = _weights

        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        if quantile_bias:
            _C = _apply_kernel(X[np.random.randint(num_examples)], weights[a1:b1], _length, dilation, padding)
            biases[i] = np.quantile(_C, quantiles[i])
        else:
            biases[i] = np.random.uniform(BIAS_RANGE[0], BIAS_RANGE[1])

        a1 = b1

    return weights, lengths, biases, dilations, paddings
