import torch.nn as nn
from mmcv import Config
from models import build_model


class SwT(nn.Module):
    def __init__(self, cfg_pth):
        super(SwT, self).__init__()
        self.cfg = Config.fromfile(cfg_pth)
        self.swin = build_model(self.cfg)

    def forward(self, x):
        x = self.swin(x)
        return x
