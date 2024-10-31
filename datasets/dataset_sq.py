import numpy as np
from pathlib import Path
from my_utils.data_utils import sample_split, sample_label_shuffle, wgn


class SQDataGenerator:
    """
    Input the selected indexes.
    """
    def __init__(self, snr=5):
        self.snr = snr
        self.sample_len = 2048

    def data_init(self):
        sq_data_path = Path(__file__).parent / rf"sq_wgn_{self.snr}dB.npy"
        if sq_data_path.exists():
            print("Load SQ data...")
            xy = np.load(sq_data_path, allow_pickle=True).item()
            x, y = xy["x"], xy["y"]
        else:
            data_files_10 = get_SQ_dir(nc=7, speed='09')
            data_files_20 = get_SQ_dir(nc=7, speed='19')
            data_files_30 = get_SQ_dir(nc=7, speed='29')
            data_files_40 = get_SQ_dir(nc=7, speed='39')
            x1, y1 = self.get_data(data_files_10, nx_per_file=36, fs=25600, fr=9)  # 确保数据量足够下采样！
            x2, y2 = self.get_data(data_files_20, nx_per_file=36, fs=25600, fr=19)
            x3, y3 = self.get_data(data_files_30, nx_per_file=36, fs=25600, fr=29)
            x4, y4 = self.get_data(data_files_40, nx_per_file=36, fs=25600, fr=39)
            x, y = np.concatenate([x1, x2, x3, x4], axis=1), np.concatenate([y1, y2, y3, y4], axis=1)
            # [Nc,num_each_way, 1, L], [Nc, num_each_way]
            x, y = sample_label_shuffle(x, y, 1)
            new_x = np.zeros_like(x)
            for c in range(x.shape[0]):
                for n in range(x.shape[1]):
                    new_x[c, n, 0] = self.add_noise(x[c, n, 0], self.snr)
            x = new_x
            np.save(sq_data_path, {"x": x, "y": y})
            print(f"Save resampled SQ data, x: {x.shape}, y: {y.shape}")
        x, y = x.astype(np.float32), y.astype(np.int32)
        # x-(7, 144, 1, self.sample_len), y-(7, 144)

        return x, y

    def get_data(self, data_files, nx_per_file, fs, fr, normalize=True):
        n_way = len(data_files)
        all_data = []
        for i in range(n_way):
            x = load_np_data(data_files[i], 200).reshape(-1)
            d = sample_split(x, True, fs, fr, n_samples=nx_per_file,
                             n_period=4, x_length=self.sample_len)  # (N, len)
            # d = sample_split(x, False, fs, fr, n_samples=nx_per_file, n_period=4, x_length=self.sample_len)  # (N, len)
            # d = my_normalization(d) if normalize else d  # do not normalize
            all_data.append(d)
        all_data = np.stack(all_data, axis=0, dtype=np.float32)[:, :, None, :]  # (n_way, n, sample_len)
        label = np.arange(n_way, dtype=np.int32).reshape(n_way, 1)
        label = np.repeat(label, all_data.shape[1], axis=1)  # [n_way, examples]
        return all_data, label  # [Nc,num_each_way, 1, L], [Nc, num_each_way]

    def add_noise(self, x, snr=10):
        return wgn(x, snr=snr)

    def data_gen(self, flatten=False):
        x, y = self.data_init()
        if flatten:
            x, y = x.reshape(-1, self.sample_len), y.reshape(-1)

        return x, y


def get_SQ_dir(nc, severity='1', speed='39'):
    """

    :param nc: number of classes
    :param severity: '1', '2', '3' # 1, 2, 3, more and more serious
    :param speed: '09', '1', '29', '39'
    :return: list
    """
    speed = str(speed)
    severity = str(severity)

    this_dir = []
    if nc == 3:
        print(f'SQ {nc}way-{severity}-{speed}Hz')
        this_dir = [r'F:\dataset_MDA\SQ\NC\NC_' + speed + '.npy',
                    r'F:\dataset_MDA\SQ\IF\IF' + severity + '_' + speed + '.npy',
                    r'F:\dataset_MDA\SQ\OF\OF' + severity + '_' + speed + '.npy']
    elif nc == 7:
        print(f'SQ {nc}way-{speed}Hz')
        this_dir = [r'F:\dataset_MDA\SQ\NC\NC_' + speed + '.npy',
                    r'F:\dataset_MDA\SQ\IF\IF1_' + speed + '.npy',
                    r'F:\dataset_MDA\SQ\IF\IF2_' + speed + '.npy',
                    r'F:\dataset_MDA\SQ\IF\IF3_' + speed + '.npy',
                    r'F:\dataset_MDA\SQ\OF\OF1_' + speed + '.npy',
                    r'F:\dataset_MDA\SQ\OF\OF2_' + speed + '.npy',
                    r'F:\dataset_MDA\SQ\OF\OF3_' + speed + '.npy',
                    ]
    return this_dir


def load_np_data(f_path, num_data):
    return np.load(f_path)[:num_data]


if __name__ == '__main__':
    X, Y = SQDataGenerator(snr=-5).data_gen()
    print(X.shape, Y.shape)  # (7, 144, 1, 2048) (7, 144)