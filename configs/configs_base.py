
from yacs.config import CfgNode as CN


_C = CN()

_C.MODEL_SAVE_DIR = r"test_results"

# ----------------- Datasets
_C.DATASET = CN()
_C.DATASET.SQ = CN()
_C.DATASET.SQ.nc = 7
_C.DATASET.SQ.out_neurons = 7
_C.DATASET.SQ.x_len = 2048

_C.DATASET.RL = CN()
_C.DATASET.RL.nc = 8
_C.DATASET.RL.out_neurons = 4
_C.DATASET.RL.x_len = 2048

_C.DATASET.SEU = CN()
_C.DATASET.SEU.nc = 10
_C.DATASET.SEU.out_neurons = 10
_C.DATASET.SEU.x_len = 1024

# ----------------- Training setup
_C.TRAIN = CN()

_C.TRAIN.EXPERIMENT = CN()
_C.TRAIN.EXPERIMENT.USE_KFOLD = False
_C.TRAIN.EXPERIMENT.TRAIN_SPLIT_RATIO = 0.3  # primary experiment
_C.TRAIN.EXPERIMENT.N_SPLITS = 10
_C.TRAIN.EXPERIMENT.K_FOLD = 5
_C.TRAIN.EXPERIMENT.N_REPEATS = 5

_C.TRAIN.MAX_EPOCHS = 40
_C.TRAIN.BATCH_SIZE = 32  # default 32
_C.TRAIN.LEARNING_RATE = 0.001


# ----------------- WKN setup
_C.WKN = CN()
_C.WKN.USE_DIFF = False
_C.WKN.USE_RESNET = True
_C.WKN.USE_LAPLACE = True

# ----------------- WKN setup
_C.MWK = CN()
_C.MWK.USE_RESNET = True


# ----------------- CWT_SVM
_C.CWT_SVM = CN()
_C.CWT_SVM.WAVE_FUN = 'scipy'  # scipy, pywt
_C.CWT_SVM.CLASSIFIER = 'fc'  # fc, svm
_C.CWT_SVM.NUM_SCALES = 16

#  ----------------- TFN
_C.TFN = CN()
_C.TFN.TYPE = 'stft'  # morlet, chirplet

#  ----------------- rocket family
_C.ROCKET = CN()
_C.ROCKET.MODEL_NAME = 'ROCKET'  # ROCKET, MiniROCKET, MultiROCKET
_C.ROCKET.USE_DIFF = False  # True, False
_C.ROCKET.N_DERIVATIVE = 1
_C.ROCKET.MULTI_NUM_FEAT = 10_000

# ------------------- Our method setup, RaVEL
_C.OURS = CN()

_C.OURS.NUM_KERNELS = 250  # 250, 500

 # original: 17.68, SQ SNR=-10; 按顺序 ablation study
_C.OURS.USE_WAVELET_KERNEL = True
_C.OURS.FEATURE_NORMALIZE = True  # 26.73
_C.OURS.QUANTILE_BIAS = True  # 36.63
_C.OURS.MULTI_POOL = True  # 40.74 ; =False i.e. 2 or =True i.e. 4

_C.OURS.MULTISCALE_WAVELET = True  # 47.38
_C.OURS.MULTI_WAVELETS = True  # 49.50

_C.OURS.DEEP_CLASSIFIER = True  # 49.79, FC classifiers
_C.OURS.N_FC = 2

_C.OURS.COSINE_POOL = True  # 52.90
_C.OURS.USE_DIFF = True  # True, False
_C.OURS.N_DERIVATIVE = 3  # 1, 3; 74.12; 82.18
_C.OURS.POOL_AFTER_COSINE = False

_C.OURS.CNN_CLASSIFIER = False
# print("********** USE CNN Classifier **********")
_C.OURS.SVM_CLASSIFIER = False
# print("********** USE SVM Classifier **********")
