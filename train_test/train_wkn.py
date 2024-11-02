import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import os
import torch

from my_utils.init_utils import set_seed_torch
from models.network.Laplace_Resnet import Laplace_ResNet
from models.network.Laplace_Inception import LaplaceInception
from models.engine.trainer_base import BaseTrainer


class NetTrainer(BaseTrainer):
    def __init__(self, dataset_name, snr, cfg_file):
        super().__init__(dataset_name=dataset_name, snr=snr, cfg_file=cfg_file)

    def construct_model(self, **kwargs):
        use_laplace = self.cfg.WKN.USE_LAPLACE
        # model = MyWKN(num_out_neurons, use_laplace, bn=True).cuda()  # 70+%
        # model = LaplaceInception(num_out_neurons, use_laplace).cuda()  # ~80%
        model = Laplace_ResNet(self.num_out_neurons if self.ml else self.num_classes,
                               use_laplace).cuda()  # best 85-90+%
        self.model_name = model.name

        return model  # best 85-90+%

    def train(self, rl_ml, one_trial):
        set_seed_torch(42, 'DCNN_train')
        self.ml = rl_ml if self.dataset_name == 'rl' else False

        # prepare data:
        self.prepare_data()

        self.DataNormalize(self.cfg.WKN.USE_DIFF, False if self.dataset_name == 'rl' else True)
        # self.DataNormalize(self.cfg.WKN.USE_DIFF, True)
        self.labels = self.labels_ml if self.ml else self.labels

        self.generate_data_indexes()
        if one_trial:
            self.train_test_indices = [self.train_test_indices[0]]
        self.k_fold_train(self.determine_start_trial())


if __name__ == "__main__":
    trainer = NetTrainer(dataset_name='sq', snr=-5, cfg_file=None)
    # trainer = NetTrainer(dataset_name='seu', snr=-5, snr=10)

    trainer.train(rl_ml=True, one_trial=True)