
# from sklearn.preprocessing import StandardScaler, normalize, maxabs_scale
import torch
import numpy as np
import random
import os
import torch.nn as nn


def set_seed_torch(seed: int = 0, option_name: str = 'Trainer'):
    print(f"======== seed {seed} for {option_name} ========")
    random.seed(seed)
    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法


# def weights_init(m):
#     if isinstance(m, nn.Conv1d):
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif isinstance(m, nn.BatchNorm1d):
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0)


def weights_init(m):
    # xavier在tanh,sigmoid中表现的很好，但在Relu激活函数中表现的很差, 何凯明提出了针对于relu的初始化方法
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data, 0., 0.02)  # generally choose this, 0.02.
        # nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    # elif isinstance(m, nn.Linear):
    #     # print(m)
    #     nn.init.normal_(m.weight, 0., 0.02)  # std=0.5 vs std=0.02, 0.02 generally. better.
        # nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu'). good.


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

