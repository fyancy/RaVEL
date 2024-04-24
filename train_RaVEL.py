"""
2023-9-23, Yong Feng, at XJTU.
The editable version of RaVEL for Ablation Study.
"""

import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from torch.utils.data import DataLoader, TensorDataset
from numba_progress import ProgressBar

from fault_diagnosis.RaVEL_ours.engine.trainer import do_train_single_label_cpu, \
    do_train_multilabel_cpu, do_train_single_label_gpu, do_train_multilabel_gpu
from fault_diagnosis.Multilabel.engine.trainer_base import BaseTrainer
from fault_diagnosis.dataset.dataset_fn import InfiniteDataset, InfiniteDataLoader


# ================== Ablation Study Start ==================
# # original: 17.68, SQ SNR=-10; 按顺序 ablation study
USE_WAVELET_KERNEL = True
FEATURE_NORMALIZE = True  # 26.73
QUANTILE_BIAS = True  # 36.63
MULTI_POOL = True  # 40.74 ; =False i.e. 2 or =True i.e. 4

MULTISCALE_WAVELET = True  # 47.38
MULTI_WAVELETS = True  # 49.50

DEEP_CLASSIFIER = True  # 49.79
print("********** USE FC Classifier **********\n")

COSINE_POOL = True  # 52.90
N_DERIVATIVE = 3  # False, 1, 3; 74.12; 82.18
POOL_COSINE = False

CNN_CLASSIFIER = False
# print("********** USE CNN Classifier **********")
SVM_CLASSIFIER = False
# print("********** USE SVM Classifier **********")

# ================== Ablation Study End ==================


# # RaVEL params:
# USE_WAVELET_KERNEL = True
# FEATURE_NORMALIZE = True
# MULTISCALE_WAVELET = True  # kernel length: (7-11) --> (11-33) --> (11-121)
# MULTI_WAVELETS = True  # only Laplace
# N_DERIVATIVE = 3  # 1, 3
# DEEP_CLASSIFIER = True
# CNN_CLASSIFIER = False
# SVM_CLASSIFIER = False
# QUANTILE_BIAS = True
# MULTI_POOL = True  # =False i.e. 2 or =True i.e. 4

# COSINE_POOL = True
# POOL_COSINE = False


class NetTrainer(BaseTrainer):
    def __init__(self, dataset_name, snr=5):
        super().__init__(dataset_name=dataset_name, snr=snr)

    def train_ravel(self, max_epochs, save_path, ml=True):
        """

        :param max_epochs:
        :param save_path:
        :param ml: only works for RL dataset. Whether to use Multilabel setup for RL dataset
        :return:
        """
        self.set_random_seed(0)
        ml = ml if self.dataset_name == 'rl' else False
        batch_size = 32

        from fault_diagnosis.RaVEL_ours.network.kernel_fit import fit_kernels
        from fault_diagnosis.RaVEL_ours.network.kernel_apply import apply_kernels_two,\
            apply_kernels_multi, apply_kernels_multi_pool_cos

        self.construct_dataset(build_train=True, build_test=True, ood_mode=False)
        t0 = time.time()
        if not self.dataset_name == 'rl':
            if N_DERIVATIVE:
                (X_tra, Y_tra), (X_val, Y_val), (X_tra_d, X_val_d) = self.DataNormalize(True, n_diff=N_DERIVATIVE)
            else:
                (X_tra, Y_tra), (X_val, Y_val) = self.DataNormalize(False)
        else:
            if N_DERIVATIVE:
                (X_tra, Y_tra), (X_val, Y_val), (X_tra_d, X_val_d), (Y_tra_sc, Y_val_sc) = \
                    self.DataNormalize(True, True, n_diff=N_DERIVATIVE)
            else:
                (X_tra, Y_tra), (X_val, Y_val), (Y_tra_sc, Y_val_sc) = self.DataNormalize(False, True)

            if not ml:
                Y_tra, Y_val = Y_tra_sc, Y_val_sc

        num_kernels = 250  # 1000, 500, 400, 250, 200,
        # todo NOTE: if num_K is smaller e.g. 250, the acc of training set will fluctuate between 95-100.
        # 1) Generate random features for ts
        kernels = fit_kernels(X_tra, num_kernels=num_kernels, multiscale=MULTISCALE_WAVELET,
                              multiwavelet=MULTI_WAVELETS, quantile_bias=QUANTILE_BIAS,
                              use_wavelet_kernel=USE_WAVELET_KERNEL)
        print("Finished kernel generation!")
        apply_kernels = apply_kernels_multi if MULTI_POOL else apply_kernels_two

        if not POOL_COSINE:
            with ProgressBar(total=len(X_tra)) as progress:
                X_tra_feat = apply_kernels(X_tra, kernels, progress, cosine_pool=COSINE_POOL)
            with ProgressBar(total=len(X_val)) as progress:
                X_val_feat = apply_kernels(X_val, kernels, progress, cosine_pool=COSINE_POOL)
        else:
            with ProgressBar(total=len(X_tra)) as progress:
                X_tra_feat = apply_kernels_multi_pool_cos(X_tra, kernels, progress)
            with ProgressBar(total=len(X_val)) as progress:
                X_val_feat = apply_kernels_multi_pool_cos(X_val, kernels, progress)

        # 2) Generate random features for diff ts
        if N_DERIVATIVE:
            kernels_d = fit_kernels(X_tra_d, num_kernels=num_kernels, multiscale=MULTISCALE_WAVELET,
                                    multiwavelet=MULTISCALE_WAVELET, quantile_bias=QUANTILE_BIAS,
                                    use_wavelet_kernel=USE_WAVELET_KERNEL)
            print("Finished kernel generation for diff series!")
            if not POOL_COSINE:
                with ProgressBar(total=len(X_tra_d)) as progress:
                    X_tra_feat_d = apply_kernels(X_tra_d, kernels_d, progress, cosine_pool=COSINE_POOL)
                with ProgressBar(total=len(X_val_d)) as progress:
                    X_val_feat_d = apply_kernels(X_val_d, kernels_d, progress, cosine_pool=COSINE_POOL)
            else:
                with ProgressBar(total=len(X_tra_d)) as progress:
                    X_tra_feat_d = apply_kernels_multi_pool_cos(X_tra_d, kernels_d, progress)
                with ProgressBar(total=len(X_val_d)) as progress:
                    X_val_feat_d = apply_kernels_multi_pool_cos(X_val_d, kernels_d, progress)
            X_tra_feat = np.concatenate((X_tra_feat, X_tra_feat_d), 1)
            X_val_feat = np.concatenate((X_val_feat, X_val_feat_d), 1)

        if FEATURE_NORMALIZE:
            f_mean, f_std = X_tra_feat.mean(0), X_tra_feat.std(0) + 1e-8  # 1e-8, 1e-12
            X_tra_feat = (X_tra_feat - f_mean) / f_std
            X_val_feat = (X_val_feat - f_mean) / f_std

        if not SVM_CLASSIFIER:
            X_tra_feat, Y_tra = torch.FloatTensor(X_tra_feat), torch.LongTensor(Y_tra)
            X_val_feat, Y_val = torch.FloatTensor(X_val_feat), torch.LongTensor(Y_val)

            if CNN_CLASSIFIER:
                X_tra_feat, X_val_feat = X_tra_feat[:, None], X_val_feat[:, None]
                valid_loader = DataLoader(TensorDataset(X_val_feat, Y_val), batch_size=min(len(Y_val), 200),
                                          shuffle=False)
                valid_loader_inf = InfiniteDataLoader(InfiniteDataset(X_val_feat, Y_val),
                                                      batch_size=min(len(Y_val), 200), shuffle=True, drop_last=True)
            train_set = TensorDataset(X_tra_feat, Y_tra)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        t1 = time.time()
        print(f"Num_kernel: {num_kernels}*2, Transformed features ({t1 - t0:.2f} s),"
              f" train-{X_tra_feat.shape}, valid-{X_val_feat.shape}")
        plot_name = f"RaVEL_diff_Nk{num_kernels}*2_snr{self.snr}" if N_DERIVATIVE else \
            f"RaVEL_Nk{num_kernels}_snr{self.snr}"

        model = self.build_classifier()

        if not SVM_CLASSIFIER:
            if self.dataset_name == 'rl' and ml:
                loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
            else:
                loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-8, patience=5)
            vis = visdom.Visdom(env='ROCKET')

            t0_train = time.time()
            if self.dataset_name == 'rl' and ml:
                if CNN_CLASSIFIER:
                    do_train_multilabel_gpu(model, optimizer,
                                            loss_fn, train_loader, valid_loader, valid_loader_inf,
                                            vis, max_epochs, save_path, scheduler, plot_name,
                                            self.num_faults, self.num_faults_train)
                else:
                    do_train_multilabel_cpu(model, optimizer, loss_fn, train_loader, (X_val_feat, Y_val),
                                            vis, max_epochs, save_path, scheduler, plot_name,
                                            self.num_faults, self.num_faults_train)
            else:
                if CNN_CLASSIFIER:
                    do_train_single_label_gpu(model, optimizer,
                                              loss_fn, train_loader, valid_loader, valid_loader_inf,
                                              vis, max_epochs, save_path, scheduler, plot_name)
                else:
                    do_train_single_label_cpu(model, optimizer, loss_fn, train_loader, (X_val_feat, Y_val),
                                              vis, max_epochs, save_path, scheduler, plot_name)

            t1_train = time.time()
            print(f"Totally: {t1_train - t0_train + t1 - t0:.2f} s, Training ({t1_train - t0_train:.2f} s), "
                  f"Trans feats ({t1 - t0:.3f} s)")
        else:
            t0_train = time.time()
            self.train_svm(X_tra_feat, Y_tra, X_val_feat, Y_val)
            t1_train = time.time()
            print(f"Totally: {t1_train - t0_train + t1 - t0:.2f} s, Training ({t1_train - t0_train:.2f} s), "
                  f"Trans feats ({t1 - t0:.3f} s)")

    def train_svm(self, X_tra, Y_tra, X_val, Y_val):
        from sklearn.svm import SVC
        from sklearn.multiclass import OneVsRestClassifier

        svc = SVC(C=2., kernel='rbf')  # rbf
        model = OneVsRestClassifier(svc)  # For Multi-class and Multi-label tasks.
        model.fit(X_tra, Y_tra)

        train_score = model.score(X_tra, Y_tra)  # the strict acc or subset acc for multilabel.
        print("Train acc: {:.2%}".format(train_score))
        test_score = model.score(X_val, Y_val)
        print("Test acc: {:.2%}".format(test_score))

    def build_classifier(self):
        use_one_fc = (not DEEP_CLASSIFIER) and (not SVM_CLASSIFIER) and (not CNN_CLASSIFIER)
        if use_one_fc:
            model = nn.Sequential(nn.LazyLinear(self.num_out_neurons))  # generally
        elif DEEP_CLASSIFIER:
            model = nn.Sequential(nn.LazyLinear(128), nn.ReLU6(),  # nn.Dropout(0.3),
                                  nn.LazyLinear(self.num_out_neurons))
            # model = nn.Sequential(nn.LazyLinear(256), nn.ReLU6(),
            #                       nn.LazyLinear(128), nn.ReLU6(),
            #                       nn.LazyLinear(self.num_classes))
        elif SVM_CLASSIFIER:
            model = None
        elif CNN_CLASSIFIER:
            from fault_diagnosis.WKN.network.wkn import SimpleCNN
            # model = Laplace_ResNet(num_classes=self.num_classes, use_laplace=False, first_conv=True)
            model = SimpleCNN(num_classes=self.num_out_neurons, bn=True, dropout=False)
        else:
            model = None
            exit("CLASSIFIER name error")

        return model


if __name__ == "__main__":
    trainer = NetTrainer(dataset_name='sq', snr=-10)  # 10, 5, 0, -5, -10, -20
    # trainer = NetTrainer(dataset_name='seu', snr=-10)
    # trainer = NetTrainer(dataset_name='rl', snr=-10)

    trainer.train_ravel(max_epochs=20, save_path=None, ml=True)
