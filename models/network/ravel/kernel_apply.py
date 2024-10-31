
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def apply_kernels_two(X, kernels, cosine_pool):
    weights, lengths, biases, dilations, paddings = kernels

    num_examples, _ = X.shape
    num_kernels = len(lengths)

    num_feat_per_kernel = 2
    _X = np.zeros((num_examples, num_kernels * num_feat_per_kernel), dtype=np.float32)  # 2 features per kernel

    for i in prange(num_examples):
        a1 = 0  # index for weights
        a2 = 0  # index for features

        for j in prange(num_kernels):
            b1 = a1 + lengths[j]
            b2 = a2 + num_feat_per_kernel

            _X[i, a2:b2] = apply_kernel_ppv_max(
                X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j])

            a1 = b1
            a2 = b2

        # progress_proxy.update(i)

    if cosine_pool:
        _X = np.cos(_X)

    return _X


@njit(fastmath=True, cache=True)
def apply_kernel_ppv_max(X, weights, length, bias, dilation, padding):
    input_length = len(X)
    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)
    end = (input_length + padding) - ((length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

    for i in range(-padding, end):
        _sum = -bias
        index = i

        for j in range(length):
            if -1 < index < input_length:
                _sum = _sum + weights[j] * X[index]
            index = index + dilation

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return _ppv / output_length, _max


# --------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def apply_kernels_multi(X, kernels, cosine_pool):
    weights, lengths, biases, dilations, paddings = kernels

    num_examples, _ = X.shape
    num_kernels = len(lengths)

    num_feat_per_kernel = 4
    _X = np.zeros((num_examples, num_kernels * num_feat_per_kernel), dtype=np.float32)  # 2 features per kernel

    for i in prange(num_examples):
        a1 = 0  # index for weights
        a2 = 0  # index for features

        for j in prange(num_kernels):
            b1 = a1 + lengths[j]
            b2 = a2 + num_feat_per_kernel

            _X[i, a2:b2] = apply_kernel_multi(
                X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j])

            a1 = b1
            a2 = b2

        # progress_proxy.update(i)

    if cosine_pool:
        _X = np.cos(_X)

    return _X


@njit(fastmath=True, cache=True)
def apply_kernel_multi(X, weights, length, bias, dilation, padding):
    input_length = len(X)
    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)
    end = (input_length + padding) - ((length - 1) * dilation)

    _ppv = 0  # PPV, proportion of positive values
    _mean = 0
    _sum_all = 0
    _max = np.NINF
    last_val = 0
    _max_stretch = 0.0  # LSPV, longest stretch of positive values
    _mean_index = 0  # MIPV, mean of indices of positive values

    for i in prange(-padding, end):  # -padding ~ length+padding
        _sum = -bias  # the difference with ROCKET, which defines "_sum = bias"
        index = i

        for j in prange(length):
            if -1 < index < input_length:
                _sum = _sum + weights[j] * X[index]  # / length NOT RECOMMEND!
            index = index + dilation

        if _sum >= 0:
            _ppv += 1
            _mean_index += i
            _mean += _sum
        else:
            stretch = i - last_val
            if stretch > _max_stretch:
                _max_stretch = stretch
            last_val = i

        # if _sum > _max:
        #     _max = _sum

        # _sum_all += _sum

    stretch = output_length - 1 - last_val
    if stretch >= _max_stretch:
        _max_stretch = stretch

    ppv = _ppv / output_length  # 84.7%
    mean1 = _mean / _ppv if _ppv > 0 else 0  # 98.44%
    mean2 = _mean / output_length if _ppv > 0 else 0  # 98.16%
    max_stretch = _max_stretch / _ppv if _ppv > 0 else 0  # 49%; 72% (/_ppv)
    # max_stretch = _max_stretch / output_length if _ppv > 0 else 0  # worse
    # mean_index = _mean_index / output_length if _ppv > 0 else -1  # 78.07%

    return ppv, max_stretch, mean2, mean1  # best, 70%+ on SQ SNR -10
    # return np.cos(ppv), np.cos(max_stretch), np.cos(mean2), np.sin(mean1)  # best,
    # return ppv, 0., np.cos(mean2), np.cos(mean1)  # best, on SEU SNR 0
    # return ppv, 0., mean1, mean2  # 87.8%
    # return ppv, mean2, 0., 0.  # original best


# --------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def apply_kernels_multi_pool_of_cos(X, kernels):
    weights, lengths, biases, dilations, paddings = kernels

    num_examples, _ = X.shape
    num_kernels = len(lengths)

    num_feat_per_kernel = 4
    _X = np.zeros((num_examples, num_kernels * num_feat_per_kernel), dtype=np.float32)  # 2 features per kernel

    for i in prange(num_examples):
        a1 = 0  # index for weights
        a2 = 0  # index for features

        for j in prange(num_kernels):
            b1 = a1 + lengths[j]
            b2 = a2 + num_feat_per_kernel

            _X[i, a2:b2] = apply_kernel_multi_pool_of_cos(
                X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j])

            a1 = b1
            a2 = b2

        # progress_proxy.update(i)

    return _X


@njit(fastmath=True, cache=True)
def apply_kernel_multi_pool_of_cos(X, weights, length, bias, dilation, padding):
    input_length = len(X)
    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)
    end = (input_length + padding) - ((length - 1) * dilation)

    _ppv = 0  # PPV, proportion of positive values
    _mean = 0
    _sum_all = 0
    _max = np.NINF
    last_val = 0
    _max_stretch = 0.0  # LSPV, longest stretch of positive values
    _mean_index = 0  # MIPV, mean of indices of positive values

    for i in prange(-padding, end):  # -padding ~ length+padding
        _sum = -bias  # the difference with ROCKET, which defines "_sum = bias"
        index = i

        for j in prange(length):
            if -1 < index < input_length:
                _sum = _sum + weights[j] * X[index]  # / length NOT RECOMMEND!
            index = index + dilation

        _sum = np.cos(_sum/length)
        if _sum >= 0:
            _ppv += 1
            _mean_index += i
            _mean += _sum
        else:
            stretch = i - last_val
            if stretch > _max_stretch:
                _max_stretch = stretch
            last_val = i

    stretch = output_length - 1 - last_val
    if stretch >= _max_stretch:
        _max_stretch = stretch

    ppv = _ppv / output_length  # 84.7%
    mean1 = _mean / _ppv if _ppv > 0 else 0  # 98.44%
    mean2 = _mean / output_length if _ppv > 0 else 0  # 98.16%
    max_stretch = _max_stretch / _ppv if _ppv > 0 else 0  # 49%; 72% (/_ppv)
    # max_stretch = _max_stretch / output_length if _ppv > 0 else 0  # for seu
    # mean_index = _mean_index / output_length if _ppv > 0 else -1  # 78.07%

    return ppv, max_stretch, mean2, mean1  # best, 70%+ on SQ SNR -10

