import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from my_utils.init_utils import set_seed_torch

from models.engine.trainer_base import BaseTrainer
from models.network.mlcnn import CNN1D, CNN1Dv2


class NetTrainer(BaseTrainer):
    def __init__(self, dataset_name, snr, cfg_file):
        super().__init__(dataset_name=dataset_name, snr=snr, cfg_file=cfg_file)

    def construct_model(self, **kwargs):
        model = CNN1Dv2(self.num_classes, self.num_out_neurons, self.ml, bn=True).cuda()
        self.model_name = model.name
        return model

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
