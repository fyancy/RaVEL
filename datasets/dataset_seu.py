"""
The gearbox dataset was collected from the drivetrain dynamic simulator (DDS) shown in Fig. 8.
Two different working conditions are investigated with the rotating speed system load
set to be either 20 HZ-0V or 30 HZ-2V. The different types of faults for bearings and
gearboxes are shown in Table VI.

The dataset contains five component conditions for the bearing data and the gearbox data:
four failure types and one health state. Therefore, fault diagnosis for DDS is a 5-class classification task.
Each fault type consist of 1000 samples for training and the entire gear and
bearing training datasets are both 5000 images. Testing datasets are the same size as training datasets.
In order to test the proposed method when dealing with mixed faults, gear faults and bearing faults
are combined to form a mixture dataset including four kinds of gear failure, four kinds of bearing failure,
and one health state. Each working state contains 1000 samples for training
and 1000 samples for testing and both the training and testing dataset contain 9000 samples.

Gearbox dataset is from Southeast University, China. These data are collected from Drivetrain Dynamic Simulator. This
dataset contains 2 subdatasets, including bearing data and gear data, which are both acquired on Drivetrain Dynamics
Simulator (DDS). There are two kinds of working conditions with rotating speed - load configuration set to be 20-0
and 30-2. Within each file, there are 8rows of signals which represent: 1-motor vibration, 2,3,4-vibration of
planetary gearbox in three directions: x, y, and z, 5-motor torque, 6,7,8-vibration of parallel gear box in three
directions: x, y, and z. Signals of rows 2,3,4 are all effective.

[1] Highly-Accurate Machine Fault Diagnosis Using Deep Transfer Learning
Paper sets sample length as 1024 points, and converts it into time-frequency image 224x224x3.

4194304 points / 2048 = 2048
"""

import numpy as np
from pathlib import Path
import h5py
from my_utils.data_utils import sample_split, wgn

root1 = None  # your raw dataset path


class SEUDataGenerator:
    def __init__(self, snr=5, x_len=1024):
        self.snr = snr
        self.sample_len = x_len

        # default setting
        self.operate_speed_id = [0, 1]
        self.chn_id = [1]

    def data_init(self):
        data_path = Path(__file__).parent / rf"seu_2speeds_snr{self.snr}.npy"
        # data_path = root1 / rf"seu_2speeds_snr{self.snr}.npy"
        # print(data_path)
        if data_path.exists():
            print("Load SEU data...")
            xy = np.load(data_path, allow_pickle=True).item()
            x, y = xy["x"], xy["y"]
        else:
            print("Resampling SEU data...")
            x, y = self.get_data()  # (2, nc, 300, L, 8-channels), (2, nc, 300)

            x = x.transpose([0, 1, 2, 4, 3])  # (2, nc, 300, 8-channels, L)
            new_x = np.zeros_like(x)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    for k in range(x.shape[2]):
                        for c in range(x.shape[3]):
                            new_x[i, j, k, c] = self.add_noise(x[i, j, k, c], self.snr)
            x = new_x
            x = x[:, :, np.random.permutation(x.shape[2])]
            np.save(data_path, {"x": x, "y": y})
            print(f"Save resampled SEU data, x: {x.shape}, y: {y.shape}")

        # x-(2, nc, 300, 8-channels, L), y-((2, nc, 300)
        x, y = x[self.operate_speed_id], y[self.operate_speed_id]  # e.g. (2, nc, 300, 8, L)
        x, y = (np.transpose(x, [1, 0, 2, 3, 4]),
                np.transpose(y, [1, 0, 2]))  # ([Nc, 2, num, 8, L], [Nc, 2, num])
        x = x[:, :, :, self.chn_id]  # e.g. [Nc, 2, num, 1, L]
        x, y = x.astype(np.float32), y.astype(np.int32)

        return x, y

    @staticmethod
    def add_noise(x, snr=10):
        return wgn(x, snr=snr)

    def get_data(self):
        p1 = root1 / r"numpy_data\GearBoxDataset_load_0.h5"
        p2 = root1 / r"numpy_data\GearBoxDataset_load_1.h5"
        f1, f2 = h5py.File(p1, "r"), h5py.File(p2, "r")
        D1, D2 = f1["data"], f2["data"]
        fs, fr = 5120, 20
        n_samples_per_cls = 300
        x_length = self.sample_len
        # ['bearing_ball', 'bearing_comb', 'bearing_health', 'bearing_inner', 'bearing_outer',
        # 'gear_chipped', 'gear_health', 'gear_miss', 'gear_root', 'gear_surface']

        data1 = np.zeros([len(D1.keys()), n_samples_per_cls, x_length, 8])
        for i, k in enumerate(D1.keys()):
            d1 = D1[k][:]  # (300, 2048, 8)
            for chn in range(d1.shape[2]):
                data1[i, :, :, chn] = sample_split(d1[:, :, chn], True, fs, fr,
                                                   n_samples_per_cls, n_period=6, x_length=x_length)

        fs, fr = 5120, 30
        data2 = np.zeros([len(D1.keys()), n_samples_per_cls, x_length, 8])
        assert D1.keys() == D2.keys()
        for i, k in enumerate(D1.keys()):
            d2 = D2[k][:]  # (300, 2048, 8)
            for chn in range(d2.shape[2]):
                data2[i, :, :, chn] = sample_split(d2[:, :, chn], True, fs, fr,
                                                   n_samples_per_cls, n_period=6, x_length=x_length)
        f1.close()
        f2.close()

        label = np.arange(data1.shape[0], dtype=np.int32).reshape([-1, 1])  # (nc, 1)
        labels = np.repeat(label, n_samples_per_cls, axis=1)  # (nc, 300)
        labels = np.stack([labels, labels], 0)  # (2, nc, 300)
        data = np.stack((data1, data2), 0)  # (2, nc, 300, 2048, 8)

        return data, labels

    def data_gen(self, flatten=False):
        x, y = self.data_init()
        if flatten:
            x = x.reshape(-1, self.sample_len)
            y = y.reshape(-1)

        return x, y


if __name__ == "__main__":
    X, Y = SEUDataGenerator(snr=-5).data_gen(False)
    print(X.shape, Y.shape)  # (10, 2, 300, 1, 1024) (10, 2, 300)
