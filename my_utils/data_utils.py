
import torch
import numpy as np
from scipy.signal import resample, resample_poly
from sklearn.preprocessing import StandardScaler, normalize, maxabs_scale
from scipy.signal import stft, cwt, morlet2
from scipy.fft import fft, fftfreq, rfft, rfftfreq
# import pywt


def sample_split_resample(x, fs=25600, fr=10, n_samples=40, n_period=4, x_length=2048):
    # x-1D signal, fr-转频, fs-采频
    x = x.reshape(-1)
    points_per_period = int(fs / fr)  # e.g. 2560
    points_in_one_sample = points_per_period * n_period
    assert len(x) >= n_samples*points_in_one_sample
    assert points_in_one_sample >= int(x_length * 0.9)
    x_before = x[:n_samples * points_in_one_sample].reshape([n_samples, points_in_one_sample])
    x_after = resample_poly(x_before, x_length, points_in_one_sample, axis=1)
    # note: fs is changed as fs*(x_length/points_in_one_sample),
    # i.e. fs*(x_length/(int(fs/fr)*n_period))) /approx x_length * fr / n_period
    return x_after


def cal_resample_fs(x_length, fs, fr, n_period):
    points_per_period = int(fs / fr)  # e.g. 2560
    points_in_one_sample = points_per_period * n_period
    fs = fs * (x_length / points_in_one_sample)
    return fs


def sample_split(x, resampling=True, fs=25600, fr=10, n_samples=50, n_period=4, x_length=2048):
    if resampling:
        return sample_split_resample(x, fs, fr, n_samples, n_period, x_length)
    else:
        return x[:n_samples * x_length].reshape([n_samples, -1])


def sample_label_shuffle(data, label, dim=0):
    """

    :param data: [num, data_len, C] (dim=0) or [n_way, num, data_len, C] (dim=1)
    :param label: [num]or [n_way, num]
    :param dim: which dimension to be shuffled
    :return:
    """

    def shuffle_first_axis(x, y):
        index = [i for i in range(len(x))]
        np.random.shuffle(index)
        return x[index], y[index]

    if dim == 0:
        return shuffle_first_axis(data, label)
    elif dim == 1:
        new_d, new_l = [], []
        for i, d in enumerate(data):
            d_, l_, = shuffle_first_axis(d, label[i])
            new_d.append(d_), new_l.append(l_)
        new_d = np.asarray(new_d)
        new_l = np.asarray(new_l)
        return new_d, new_l
    else:
        raise ValueError("dim must be 0 or 1")


def my_normalization(x):  # x: (n, length)
    # method 1:
    # x = x - np.mean(x, axis=1, keepdims=True)
    # x = maxabs_scale(x, axis=1)  # 效果也很好, 2
    # method 2:
    # x = normalize(x, norm='l2', axis=1)
    # method 3: in LiftingNet
    std_scaler = StandardScaler()
    x = np.transpose(x, [1, 0])
    x = std_scaler.fit_transform(x)
    x = np.transpose(x, [1, 0])

    return np.asarray(x)


# Compute the data range
def compute_dataset_range(train_data, return_scalar=True):
    # train_data-[N, 1, L]
    data_max = -np.ones(train_data[0, 0].shape, dtype=float) * np.inf  # (L,)
    data_min = np.ones_like(data_max) * np.inf
    for i in range(len(train_data)):
        x = train_data[i][0]
        data_max = np.maximum(data_max, x)
        data_min = np.minimum(data_min, x)

    data_range = data_max - data_min
    data_range = np.max(data_range) if return_scalar else data_range
    return data_range


def wgn(x, snr):
    len_x = len(x)
    Ps = np.sum(np.power(x, 2)) / len_x
    # Pg = np.sum(np.power(gaussian_noise, 2)) / len_x  # Pg is about 1
    Pn = Ps / (np.power(10, snr / 10))
    noise = np.random.randn(len_x) * np.sqrt(Pn)  # Pn/Pg is about Pn
    return x + noise


def batch_wgn(x, snr):
    """
    作者：强劲九
    链接：https://juejin.cn/post/6971745037965066254
    来源：稀土掘金
    著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
    :param x:
    :param snr:
    :return:
    """
    batch_size, len_x = x.shape
    Ps = np.sum(np.power(x, 2)) / len_x
    # Pg = np.sum(np.power(gaussian_noise, 2)) / len_x  # Pg is about 1
    Pn = Ps / (np.power(10, snr / 10))
    noise = np.random.randn(batch_size, len_x) * np.sqrt(Pn)  # Pn/Pg is about Pn
    return x + noise


def stft_batch(x_batch, fs=1.0):
    """

    :param x_batch: a batch of data, e.g. (N, L)
    :param fs: float, the sampling frequency, but this factor does NOT affect the result.
    :return:
    """
    f, t, Zxx = stft(x_batch, fs=fs, window='hamming', nperseg=64, axis=-1)  #
    img = np.abs(Zxx)  # Zxx: (len(f), len(t)), \approx (nperseg//2+1, n_points//overlap+1)

    # The following operations are NOT recommended:
    # img = img / np.max(img, axis=(1, 2), keepdims=True) * 255.
    # img = img.astype(np.int32)  # e.g. (65, 33)
    # img = img[:, :img.shape[1]//2]

    return img


def fft_batch(x_batch, fs=1.0):
    """

    :param x_batch: a batch of data, e.g. (N, L)
    :param fs: float, the sampling frequency, but this factor does NOT affect the result.
    :return:
    """
    L = x_batch.shape[-1]
    yf = fft(x_batch, axis=-1)
    # xf = fftfreq(L, d=1.0/fs)[:L//2]
    yf = np.abs(yf[:, :L//2])
    yf = yf / np.max(yf, axis=-1, keepdims=True)

    return yf


def rfft_batch(x_batch, fs=1.0):
    """

    :param x_batch: a batch of data, e.g. (N, L)
    :param fs: float, the sampling frequency, but this factor does NOT affect the result.
    :return:
    """
    # xf = fftfreq(L, d=1.0/fs)
    yf = np.abs(rfft(x_batch, axis=-1))
    yf = yf / np.max(yf, axis=-1, keepdims=True)

    return yf


def cwt_batch(x_batch, num_scales=32, fs=1.0):
    """
    refer to: 本人CSDN已记录该函数使用过程
    https://blog.csdn.net/weixin_46713695/article/details/127234673
    https://github.com/Abhijit-Bhumireddy99/RUL_Prediction/blob/main/Test_Dataset_Preparation.ipynb
    https://pywavelets.readthedocs.io/en/latest/ref/cwt.html

    :param num_scales: number of scales, but only display 5% scales or frequencies for analysis.
    :param x_batch: a batch of data, e.g. (N, L)
    :param fs: float, the sampling frequency, but this factor does NOT affect the result.
    :return: (N, scales, L)
    """

    fc = 60.  # fc=40~60 better
    freq = np.linspace(1, fs / 2, num_scales)
    widths = fc * fs / (2 * freq * np.pi)

    feats = []
    for x in x_batch:
        # [cwtm, freqs] = pywt.cwt(x, widths, wave_name, 1 / fs, axis=-1)
        cwtm = cwt(x, morlet2, widths, w=fc)  # float64 is more precise
        feats.append(np.abs(cwtm))  # [len(scales): 50, len(t): 2048]
    feats = np.stack(feats, 0)

    return feats


def convert_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        print(type(x))
        exit("Input Type error.")


def augment(x):
    if np.random.rand() < 0.5:  # flip, horizontal
        x = x[::-1]
    if np.random.rand() < 0.3:  # flip, vertical
        x = -x
    return x




