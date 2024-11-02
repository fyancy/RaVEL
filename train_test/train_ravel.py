
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('..')

import os
import numpy as np
import time
import torch
import torch.nn as nn
import visdom

from torch.utils.data import DataLoader, TensorDataset

from models.engine.trainer_base import BaseTrainer
from my_utils.init_utils import set_seed_torch
from models.network.ravel.kernel_fit import fit_kernels
from models.network.ravel.kernel_apply import apply_kernels_two,\
    apply_kernels_multi, apply_kernels_multi_pool_of_cos
from datasets.dataset_fn import InfiniteDataset, InfiniteDataLoader
from models.engine.trainerv2 import do_train_single_label_cpu, \
    do_train_multilabel_cpu, do_train_single_label_gpu, do_train_multilabel_gpu


class RavelTrainer(BaseTrainer):
    def __init__(self, dataset_name, snr, cfg_file=None, cfg_list=None, train_log=False):
        super().__init__(dataset_name=dataset_name, snr=snr,
                         cfg_file=cfg_file, cfg_list=cfg_list, train_log=train_log)
    
    def construct_model(self, **kwargs):
        self.model_name = 'RAVEL'
    
    def build_classifier(self):
        use_one_fc = ((not self.cfg.OURS.DEEP_CLASSIFIER) and
                      (not self.cfg.OURS.SVM_CLASSIFIER) and (not self.cfg.OURS.CNN_CLASSIFIER))
        out_chns = self.num_out_neurons if self.ml else self.num_classes
        if use_one_fc:
            model = nn.Sequential(nn.LazyLinear(out_chns))  # generally
        elif self.cfg.OURS.DEEP_CLASSIFIER:
            if self.cfg.OURS.N_FC == 2:
                model = nn.Sequential(nn.LazyLinear(128), nn.ReLU6(),  # nn.Dropout(0.3),
                                      nn.LazyLinear(out_chns))
            elif self.cfg.OURS.N_FC == 3:
                model = nn.Sequential(nn.LazyLinear(256), nn.ReLU6(),
                                      nn.LazyLinear(128), nn.ReLU6(),
                                      nn.LazyLinear(out_chns))
            else:
                exit("MLP with more than 3 FC layers is not supported yet.")

        elif self.cfg.OURS.SVM_CLASSIFIER:
            model = None
        elif self.cfg.OURS.CNN_CLASSIFIER:
            from models.network.wkn import SimpleCNN
            # model = Laplace_ResNet(num_classes=self.num_classes, use_laplace=False, first_conv=True)
            model = SimpleCNN(num_classes=out_chns, bn=True, dropout=False)
        else:
            model = None
            exit("CLASSIFIER name error")

        return model

    def train(self, rl_ml, one_trial=False):
        set_seed_torch(42, f'train_{self.model_name}_{self.dataset_name}_{self.snr}')
        self.ml = rl_ml if self.dataset_name == 'rl' else False

        # prepare data:
        self.prepare_data()
        self.DataNormalize(self.cfg.OURS.USE_DIFF, True, n_diff=self.cfg.OURS.N_DERIVATIVE)
        self.labels = self.labels_ml if self.ml else self.labels

        # build model:
        self.generate_data_indexes()
        if one_trial:
            self.train_test_indices = [self.train_test_indices[0]]  # only one trial
        self.k_fold_train(self.determine_start_trial(), show_vis=False)
    
    def k_fold_train(self, start_trial=1, show_vis=True):
        use_kfold = self.cfg.TRAIN.EXPERIMENT.USE_KFOLD
        self.vis = visdom.Visdom(env='RaVEL_R1') if show_vis else None
        for i in range(len(self.train_test_indices)):
            if i > 0:
                self.vis = None
            if i + 1 < start_trial:
                continue

            self.trial_num = i + 1
            if use_kfold:
                repeat_count = i//self.cfg.TRAIN.EXPERIMENT.K_FOLD+1
                fold_count = (i+1) - (repeat_count-1)*self.cfg.TRAIN.EXPERIMENT.K_FOLD
                print("\n\n**********Repeat Test {}, Dataset Fold {} **********\n".format(
                    repeat_count, fold_count))
            else:
                print("\n\n********** Trial No. #{}/{} **********\n".format(
                    i+1, len(self.train_test_indices)))
            
            ids_train, ids_test = self.train_test_indices[i]
            Xtr, Xte = self.data[ids_train], self.data[ids_test]
            Ytr, Yte = self.labels[ids_train], self.labels[ids_test]
            if self.cfg.OURS.USE_DIFF:
                Xtr_d, Xte_d = self.data_diff[ids_train], self.data_diff[ids_test]
            else:
                Xtr_d = Xte_d = None
            print(f"train-{Xtr.shape}, {Ytr.shape}, test-{Xte.shape}, {Yte.shape}")

            # 训练模型
            self.one_trial_train(Xtr, Ytr, Xte, Yte, Xtr_d, Xte_d, plot_reliability=False)
            print(f"Trial {i+1} finished.")
        
        print("\n********** All trials finished **********\n")
        if self.logging:
            self.calculate_average_metrics(self.path_dict['test_results'])

    def one_trial_train(self, Xtr, Ytr, Xte, Yte, Xtr_d, Xte_d, plot_reliability=False):
        self.time_train, self.time_test = 0., 0.
        t0 = time.time()

        # 1) Generate random features for ts
        kernels = fit_kernels(Xtr,
                              num_kernels=self.cfg.OURS.NUM_KERNELS,
                              multiscale=self.cfg.OURS.MULTISCALE_WAVELET,
                              multiwavelet=self.cfg.OURS.MULTI_WAVELETS,
                              quantile_bias=self.cfg.OURS.QUANTILE_BIAS,
                              use_wavelet_kernel=self.cfg.OURS.USE_WAVELET_KERNEL)
        print("Finished kernel generation!")

        apply_kernels = apply_kernels_multi if self.cfg.OURS.MULTI_POOL else apply_kernels_two
        if not self.cfg.OURS.POOL_AFTER_COSINE:
            X_tra_feat = apply_kernels(Xtr, kernels, cosine_pool=self.cfg.OURS.COSINE_POOL)
            _test_t01 = time.time()
            X_val_feat = apply_kernels(Xte, kernels, cosine_pool=self.cfg.OURS.COSINE_POOL)
            self.time_test += time.time() - _test_t01
        else:
            X_tra_feat = apply_kernels_multi_pool_of_cos(Xtr, kernels)
            _test_t01 = time.time()
            X_val_feat = apply_kernels_multi_pool_of_cos(Xte, kernels)
            self.time_test += time.time() - _test_t01

        # 2) Generate random features for diff ts
        if self.cfg.OURS.USE_DIFF:
            kernels_d = fit_kernels(Xtr_d,
                                    num_kernels=self.cfg.OURS.NUM_KERNELS,
                                    multiscale=self.cfg.OURS.MULTISCALE_WAVELET,
                                    multiwavelet=self.cfg.OURS.MULTISCALE_WAVELET,
                                    quantile_bias=self.cfg.OURS.QUANTILE_BIAS,
                                    use_wavelet_kernel=self.cfg.OURS.USE_WAVELET_KERNEL)
            print("Finished kernel generation for diff series!")
            if not self.cfg.OURS.POOL_AFTER_COSINE:
                X_tra_feat_d = apply_kernels(Xtr_d, kernels_d, cosine_pool=self.cfg.OURS.COSINE_POOL)
                _test_t11 = time.time()
                X_val_feat_d = apply_kernels(Xte_d, kernels_d, cosine_pool=self.cfg.OURS.COSINE_POOL)
                self.time_test += time.time() - _test_t11
            else:
                X_tra_feat_d = apply_kernels_multi_pool_of_cos(Xtr_d, kernels_d)
                _test_t11 = time.time()
                X_val_feat_d = apply_kernels_multi_pool_of_cos(Xte_d, kernels_d)
                self.time_test += time.time() - _test_t11

            X_tra_feat = np.concatenate((X_tra_feat, X_tra_feat_d), 1)
            X_val_feat = np.concatenate((X_val_feat, X_val_feat_d), 1)

        if self.cfg.OURS.FEATURE_NORMALIZE:
            f_mean, f_std = X_tra_feat.mean(0), X_tra_feat.std(0) + 1e-8  # 1e-8, 1e-12
            X_tra_feat = (X_tra_feat - f_mean) / f_std
            _test_t21 = time.time()
            X_val_feat = (X_val_feat - f_mean) / f_std
            self.time_test += time.time() - _test_t21

        if not self.cfg.OURS.SVM_CLASSIFIER:
            X_tra_feat, Y_tra = torch.FloatTensor(X_tra_feat), torch.LongTensor(Ytr)
            X_val_feat, Y_val = torch.FloatTensor(X_val_feat), torch.LongTensor(Yte)

            if self.cfg.OURS.CNN_CLASSIFIER:
                X_tra_feat, X_val_feat = X_tra_feat[:, None], X_val_feat[:, None]
                valid_loader = DataLoader(TensorDataset(X_val_feat, Y_val),
                                          batch_size=min(len(Y_val), 200), shuffle=False)
                valid_loader_inf = InfiniteDataLoader(InfiniteDataset(X_val_feat, Y_val),
                                                      batch_size=min(len(Y_val), 200), shuffle=True, drop_last=True)
            train_set = TensorDataset(X_tra_feat, Y_tra)
            train_loader = DataLoader(train_set, batch_size=self.cfg.TRAIN.BATCH_SIZE, shuffle=True)

        t1 = time.time()
        test_feat_time = self.time_test
        train_feat_time = t1 - t0 - test_feat_time

        print(f"Num_kernel: {self.cfg.OURS.NUM_KERNELS}*2, Transformed features ({train_feat_time:.2f} s),"
              f" train-{X_tra_feat.shape}, valid-{X_val_feat.shape}")
        plot_name = f"RaVEL_diff_Nk{self.cfg.OURS.NUM_KERNELS}*2_snr{self.snr}" if self.cfg.OURS.USE_DIFF else \
            f"RaVEL_Nk{self.cfg.OURS.NUM_KERNELS}_snr{self.snr}"

        model = self.build_classifier()
        if not self.cfg.OURS.SVM_CLASSIFIER:
            loss_fn = nn.BCEWithLogitsLoss(reduction='sum') if self.ml else nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

            t0_train = time.time()
            if self.ml:
                if self.cfg.OURS.CNN_CLASSIFIER:
                    results = do_train_multilabel_gpu(model=model, optimizer=optimizer,
                                                      loss_fn=loss_fn, train_loader=train_loader,
                                                      valid_loader=valid_loader, valid_loader_inf=valid_loader_inf,
                                                      vis=self.vis, max_epochs=self.cfg.TRAIN.MAX_EPOCHS,
                                                      scheduler=scheduler, plot_name=plot_name,
                                                      num_cls=self.num_faults, num_cls_train=self.num_faults_train)
                else:
                    results = do_train_multilabel_cpu(model=model, optimizer=optimizer,
                                                      loss_fn=loss_fn, train_loader=train_loader,
                                                      valid_data=(X_val_feat, Y_val),
                                                      vis=self.vis, max_epochs=self.cfg.TRAIN.MAX_EPOCHS,
                                                      scheduler=scheduler, plot_name=plot_name,
                                                      num_cls=self.num_faults, num_cls_train=self.num_faults_train)
            else:
                if self.cfg.OURS.CNN_CLASSIFIER:
                    results = do_train_single_label_gpu(model=model, optimizer=optimizer,
                                                        loss_fn=loss_fn, train_loader=train_loader,
                                                        valid_loader=valid_loader, valid_loader_inf=valid_loader_inf,
                                                        vis=self.vis, max_epochs=self.cfg.TRAIN.MAX_EPOCHS,
                                                        scheduler=scheduler, plot_name=plot_name)
                else:
                    results = do_train_single_label_cpu(model=model, optimizer=optimizer,
                                                        loss_fn=loss_fn, train_loader=train_loader,
                                                        valid_data=(X_val_feat, Y_val),
                                                        vis=self.vis, max_epochs=self.cfg.TRAIN.MAX_EPOCHS,
                                                        scheduler=scheduler, plot_name=plot_name)
        else:
            t0_train = time.time()
            results = self.train_svm(X_tra_feat, Ytr, X_val_feat, Yte)
            t1_train = time.time()
            print(f"Totally: {t1_train - t0_train + t1 - t0:.2f} s, Training ({t1_train - t0_train:.2f} s), "
                  f"Trans feats ({t1 - t0:.3f} s)")

        t1_train = time.time()
        print(f"Train Classifier Finished ({t1_train - t0_train:.2f} s)")
        results['train_time'] = train_feat_time + t1_train - t0_train - results['test_time']
        results['test_time'] = test_feat_time + results['test_time']
        results = {k: [v] for k, v in results.items()}
        if self.logging:
            self.save_model_results(model, results)

        print("Totally: {:.2f}s, Train: {:.2f}s, Test: {:.2f}s".format(
            t1_train - t0_train + t1 - t0, results['train_time'][0], results['test_time'][0]
        ))


if __name__ == '__main__':
    cfg_path = os.path.join(os.path.pardir, r"configs\cfg_ravel.yaml")
    trainer = RavelTrainer(dataset_name='sq', snr=-5, cfg_file=cfg_path)
    # trainer = RavelTrainer(dataset_name='seu', snr=-5, cfg_file=cfg_path)
    trainer.train(rl_ml=True, one_trial=True)  # acc-ml and acc-sc
