from numba.core.ir import Return
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit
import hypertools as hyp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import visdom
import os
import time
import pandas as pd
import datetime

from my_utils.init_utils import set_seed_torch
from my_utils.data_utils import convert_to_numpy
from my_utils.figure_sets import set_figure
from datasets import SEUDataGenerator, SQDataGenerator
from datasets.dataset_fn import InfiniteDataset, InfiniteDataLoader
# from my_utils.score_utils import cal_brier_score
from configs.configs_base import _C
from models.engine.trainerv2 import do_train_single_label_gpu, do_train_multilabel_gpu
from my_utils.logger import set_logger


class BaseTrainer:
    def __init__(self, dataset_name, snr=5, seed=0,
                 cfg_file=None, cfg_list=None, train_log=False, **kwargs):
        self.set_random_seed(seed)
        self.cfg = self.setup(_C, cfg_file, cfg_list)
        # construct attributes by kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        nc = out_neurons = 4
        x_len = 2048
        if dataset_name == 'sq':
            nc = out_neurons = self.cfg.DATASET.SQ.nc
            x_len = self.cfg.DATASET.SQ.x_len
        elif dataset_name == 'seu':
            nc = out_neurons =  self.cfg.DATASET.SEU.nc
            x_len = self.cfg.DATASET.SEU.x_len
        elif dataset_name == 'rl':
            nc = self.cfg.DATASET.RL.nc
            out_neurons = self.cfg.DATASET.RL.out_neurons
            x_len = self.cfg.DATASET.RL.x_len
        else:
            exit("Dataset Name Error.")

        self.x_len = x_len
        self.num_classes = nc
        self.num_faults = self.num_classes
        self.num_faults_train = self.num_classes
        self.num_out_neurons = out_neurons

        self.dataset_name = dataset_name
        self.model_name = 'RaVEL'
        self.snr = snr  # 10 ~ -10

        self.data = None
        self.data_diff = None
        self.labels = None
        self.labels_sc = None
        self.labels_ml = None
        self.train_test_indices = None
        self.ml = None  # use multilabel, only support rl dataset.
        self.trial_num = 0
        self.vis = None
        self.time_train = 0.
        self.time_test = 0.

        # ------------- Now start the log
        self.construct_model()  # get model name
        self.logging = train_log
        self.path_dict = {'test_results': '', 'logs': ''}
        if self.logging:
            _path = os.path.join(self.cfg.MODEL_SAVE_DIR, rf"{self.model_name}")
            os.makedirs(_path, exist_ok=True)
            log_path = os.path.join(_path, rf"{self.model_name}_{self.dataset_name}_SNR{self.snr}_log.txt")
            set_logger(filename=log_path)
            self.path_dict['logs'] = log_path
            print(f"Trainer: {self.model_name} on {self.dataset_name} dataset SNR={self.snr}")
            print(datetime.datetime.now(), '\n')

    @staticmethod
    def setup(cfg, cfg_file, cfg_list=None):
        if cfg_file is not None:
            cfg.merge_from_file(cfg_file)  # .yaml
        if cfg_list is not None:  # e.g. ['TFN.TYPE', 'chirplet']
            cfg.merge_from_list(cfg_list)
        cfg.freeze()

        return cfg

    @staticmethod
    def set_random_seed(seed=0):
        set_seed_torch(seed, 'Trainer')

    def DataNormalize(self, diff=False, normalize=True, n_diff=3):
        assert self.data.ndim == 2
        raw_data = self.data.copy()

        if normalize:
            self.data =  ((raw_data - raw_data.mean(1, keepdims=True)) /
                          (raw_data.std(1, keepdims=True) + 1e-12))
        # std or norm2 are both OK
        # X_tra_norm = norm(X_tra, axis=1)
        # X_val_norm = norm(X_val, axis=1)  # both OK

        if diff:  # NOTE: 不能在归一化基础上差分，不能在差分基础上归一化，否则效果很差
            if isinstance(diff, str) and diff.lower() == 'even':
                # get the even points
                self.data_diff = np.diff(raw_data[:, ::2], n_diff)
            else:
                # data_diff = np.diff(np.abs(raw_data), n_diff)  # worse
                self.data_diff = np.diff(raw_data, n_diff)
            # X_tra_d = (X_tra_d - X_tra_d.mean(1, keepdims=True)) / X_tra_d.std(1, keepdims=True)  # diff归一化效果差
            # X_val_d = (X_val_d - X_val_d.mean(1, keepdims=True)) / X_val_d.std(1, keepdims=True)

    def plot_features(self, feats, nc=10, fig_path=None):
        feats = convert_to_numpy(feats)  # (N, d1, d2, ..., dN)
        feats = feats.reshape(len(feats), -1)
        num_per_cls = len(feats) // nc
        labels = np.arange(nc).repeat(num_per_cls)  # (nc*num_per_cls)

        legend = ["NC", "IF1", "IF2", "IF3", "OF1", "OF2", "OF3"] if self.dataset_name == 'sq' else None
        set_figure(font_size=8, tick_size=7, ms=5., fig_w=6, hw_ratio=1, lw=0.1)
        # plt.style.use('ggplot')
        # hyp.plot(feats, '.', reduce='TSNE', ndims=2, hue=labels, palette='Spectral',
        #          labels=self.get_labels_string(len(feats), nc), size=(6 / 2.54, 6 / 2.54), save_path=fig_path
        #          )
        hyp.plot(feats, '.', reduce='TSNE', ndims=2, hue=labels, palette='Spectral',
                 size=(6 / 2.54, 6 / 2.54), save_path=fig_path, legend=legend
                 )
        plt.show()

    def get_labels_string(self, num_samples, nc):
        dataset_name = self.dataset_name
        if dataset_name == 'sq':
            n_per_cls = num_samples // nc
            un_labels = n_per_cls - 1
            labeling = ['NC'] + [None] * un_labels + \
                       ['IF1'] + [None] * un_labels + \
                       ['IF2'] + [None] * un_labels + \
                       ['IF3'] + [None] * un_labels + \
                       ['OF1'] + [None] * un_labels + \
                       ['OF2'] + [None] * un_labels + \
                       ['OF3'] + [None] * un_labels
        else:
            labeling = None
            print(f"{dataset_name} dataset are NOT supported for labeling.")

        return labeling

    def prepare_data(self, flatten=True):
        print("Prepare Data: {}, SNR={}".format(self.dataset_name, self.snr))

        if self.dataset_name == 'rl':  # may be multi-label
            raise NotImplementedError("RL dataset is not supported.")
            # self.data, self.labels, self.labels_ml = RLDataGenerator(
            #     snr=self.snr, x_len=self.x_len).data_gen(flatten)
        elif self.dataset_name == 'sq':
            self.data, self.labels = SQDataGenerator(snr=self.snr).data_gen(flatten)
        elif self.dataset_name == 'seu':
            self.data, self.labels = SEUDataGenerator(snr=self.snr, x_len=self.x_len).data_gen(flatten)
        else:
            exit("Dataset Name Error.")

        self.labels_sc = self.labels.copy()
        print("Data: {}, Labels-sc: {}, Labels-ml: {}".format(
            self.data.shape, self.labels_sc.shape,
            self.labels_ml.shape if self.labels_ml is not None else None))

    def generate_data_indexes(self):
        assert self.data is not None

        if self.cfg.TRAIN.EXPERIMENT.USE_KFOLD:
            k = self.cfg.TRAIN.EXPERIMENT.K_FOLD
            n = self.cfg.TRAIN.EXPERIMENT.N_REPEATS
            # kf = RepeatedKFold(n_splits=k, n_repeats=n, random_state=42)
            kf = RepeatedStratifiedKFold(n_splits=k, n_repeats=n, random_state=42)  # 按类别比例抽样
            self.train_test_indices = list(kf.split(self.data, self.labels_sc))
            # shape: (n_splits*n_repeats, train: (1-1/K)*num_samples, test: 1/K*num_samples)
        else:
            kf = StratifiedShuffleSplit(n_splits=self.cfg.TRAIN.EXPERIMENT.N_SPLITS,
                                        train_size=self.cfg.TRAIN.EXPERIMENT.TRAIN_SPLIT_RATIO,
                                        random_state=42)  # 按类别比例抽样
            self.train_test_indices = list(kf.split(self.data, self.labels_sc))

        print("Sample indices to train and test are obtained.")

    def k_fold_train(self, start_trial=1, show_vis=True):
        use_kfold = self.cfg.TRAIN.EXPERIMENT.USE_KFOLD
        self.vis = None  # visdom.Visdom(env='RaVEL') if show_vis else None
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
                print("\n\n********** Trial No. #{}/{} **********\n".format(i+1, len(self.train_test_indices)))

            ids_tra, ids_test = self.train_test_indices[i]
            Xtr, Xte = self.data[ids_tra], self.data[ids_test]
            Ytr, Yte = self.labels[ids_tra], self.labels[ids_test]
            Xtr, Ytr = torch.FloatTensor(Xtr[:, None]), torch.LongTensor(Ytr)
            Xte, Yte = torch.FloatTensor(Xte[:, None]), torch.LongTensor(Yte)
            print(f"train-{Xtr.shape}, {Ytr.shape}, test-{Xte.shape}, {Yte.shape}")

            self.one_time_train(Xtr, Ytr, Xte, Yte, plot_reliability=False)
            print("Trial {} finished.".format(i+1))

        print("\n*********************************")
        print("All trials finished.\n")
        self.calculate_average_metrics(self.path_dict['test_results'])

    def one_time_train(self, Xtr, Ytr, Xte, Yte, Xtr_d=None, Xte_d=None, plot_reliability=False):
        model = self.construct_model()
        loss_fn = nn.BCEWithLogitsLoss(reduction='sum') if self.ml else nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        train_loader = DataLoader(TensorDataset(Xtr, Ytr),
                                  batch_size=self.cfg.TRAIN.BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(TensorDataset(Xte, Yte),
                                  batch_size=min(len(Yte), 200), shuffle=False)
        valid_loader_inf = InfiniteDataLoader(InfiniteDataset(Xte, Yte),
                                              batch_size=min(len(Yte), 200), shuffle=True, drop_last=True)

        plot_name = "{}_{}_{}_SNR{}".format(
            self.model_name, "ML" if self.ml else "SC", self.dataset_name, self.snr)

        path0 = rf"E:\PhD Docs\MyWork\第10篇_2023随机特征\数据分析\figs\诊断结果置信度\{plot_name}.pth"
        if plot_reliability and os.path.exists(path0):
            model.load_state_dict(torch.load(path0))
            self.plot_reliability(model, valid_loader, self.ml, plot_name)
            exit()

        t0_train = time.time()
        if self.ml:
            results = do_train_multilabel_gpu(model=model, optimizer=optimizer,
                                              loss_fn=loss_fn, train_loader=train_loader, valid_loader=valid_loader,
                                              valid_loader_inf=valid_loader_inf,
                                              vis=self.vis, max_epochs=self.cfg.TRAIN.MAX_EPOCHS,
                                              scheduler=scheduler, plot_name=plot_name,
                                              num_cls=self.num_faults, num_cls_train=self.num_faults_train)
        else:
            results = do_train_single_label_gpu(model=model, optimizer=optimizer,
                                                loss_fn=loss_fn, train_loader=train_loader,
                                                valid_loader=valid_loader, valid_loader_inf=valid_loader_inf,
                                                vis=self.vis, max_epochs=self.cfg.TRAIN.MAX_EPOCHS,
                                                scheduler=scheduler, plot_name=plot_name)
        t1_train = time.time()
        print(f"Training Finished ({t1_train - t0_train:.2f} s)")
        results['train_time'] = \
            self.time_train + t1_train - t0_train - results['test_time'] - results['test_time_cpu']
        results['test_time'] = self.time_test + results['test_time']
        results['test_time_cpu'] = self.time_test + results['test_time_cpu']
        results = {k: [v] for k, v in results.items()}
        self.save_model_results(model, results)

        if plot_reliability and not os.path.exists(path0):
            torch.save(model.state_dict(), path0)
            print(f"**************\nSave model in [{path0}]")

        if plot_reliability:
            self.plot_reliability(model, valid_loader, self.ml, plot_name)

    def construct_model(self, **kwargs):
        self.model_name = None

    def plot_reliability(self, model, valid_loader, ml, plot_name):

        model.eval()
        val_logits = []
        val_ys = []
        with torch.no_grad():
            for bx, by in valid_loader:
                bx, by = bx.cuda(), by.cuda().long()
                _logits = model(bx)
                val_logits.append(_logits)
                val_ys.append(by)
        model.train()
        val_logits = torch.cat(val_logits, 0)
        val_ys = torch.cat(val_ys, 0)

        if self.dataset_name == 'rl' and ml:
            fig = self.reliability_plot_binary_multilabel(val_logits, val_ys)
        else:
            fig = self.reliability_plot_multiclass(val_logits, val_ys)

        save_path = r"E:\PhD Docs\MyWork\第10篇_2023随机特征\数据分析\figs\诊断结果置信度"
        fig_name = rf"\{plot_name}.pdf"
        if not os.path.exists(save_path + fig_name):
            fig.savefig(save_path + fig_name, bbox_inches='tight')

    def load_model(self, model):
        _path = os.path.join(self.cfg.MODEL_SAVE_DIR, rf"{self.model_name}")
        model_path = os.path.join(
            _path, rf"{self.model_name}_{self.dataset_name}_SNR{self.snr}_trial_{self.determine_start_trial()-1}.pth")
        model.load_state_dict(torch.load(model_path))

        return model

    def save_model_results(self, model, results):
        if not os.path.exists(self.cfg.MODEL_SAVE_DIR) or not self.logging:
            return

        _path = os.path.join(self.cfg.MODEL_SAVE_DIR, rf"{self.model_name}")
        os.makedirs(_path, exist_ok=True)

        if model is not None:
            model_path = os.path.join(
                _path, rf"{self.model_name}_{self.dataset_name}_SNR{self.snr}_trial_{self.trial_num}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"**************\nSave model in [{model_path}]")

        results_path = os.path.join(_path, rf"{self.model_name}_{self.dataset_name}_SNR{self.snr}.csv")
        df = pd.DataFrame(results)
        self.save_results(results_path, df)

        if self.path_dict['test_results'] is None:
            self.path_dict['test_results'] = results_path

    def determine_start_trial(self):
        # trial number
        _path = os.path.join(self.cfg.MODEL_SAVE_DIR, rf"{self.model_name}")
        if not os.path.exists(_path) or not self.logging:
            start_trial_num = 1
        else:
            files = os.listdir(_path)
            # select files start with self.model_name}_{self.dataset_name}
            files = [f for f in files if f.startswith(self.model_name + '_' + self.dataset_name)]
            pth_files = [f for f in files if f.endswith('.pth')]
            if len(pth_files) == 0:
                start_trial_num = 1
            else:
                start_trial_num = max([int(f.split('_')[-1].split('.')[0]) for f in pth_files]) + 1

        print('start_trial_num: ', start_trial_num)
        return start_trial_num

    def save_results(self, save_file_path, new_df):
        if not os.path.exists(save_file_path):
            new_df.to_csv(save_file_path, index=False)
        else:
            _df = pd.read_csv(save_file_path)
            new_df = pd.concat([_df, new_df])
            new_df.to_csv(save_file_path, index=False)
            print(f"**************\nSave results in [{save_file_path}]")

    def calculate_average_metrics(self, csv_file_path, drop_first=True):
        if not os.path.exists(csv_file_path) or not self.logging:
            return
        df = pd.read_csv(csv_file_path)
        # if drop_first:
        #     df = df.drop(0, axis=0)
        # if you use visdom for visualization, plz delete the first row, visdom really takes time.
        print("----++++---- Mean: \n {} \n ----++++---- Std: \n {} ".format(
            df.mean(), df.std()
        ))


if __name__ == '__main__':
    _cfg = _C.clone()
    print(_cfg.DATASET.SQ.nc)