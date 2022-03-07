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


def conv_block(in_f, out_f, activation='relu', *args, **kwargs):
    activations = nn.ModuleDict({
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
        'none': nn.Identity()
    })

    conv = nn.Conv2d

    return nn.Sequential(
        conv(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f, affine=True),
        activations[activation],
    )


def res_encoder(in_c):
    encoder = nn.Sequential(
        conv_block(in_c, 16, kernel_size=7, padding=3, stride=1),

        # ResNetBlock(16, 16),

        # ResNetBlock(16, 16),
        # nn.MaxPool2d(kernel_size=1),
        # # nn.Dropout(p=0.25),

        ResNetBlock(16, 32),
        nn.MaxPool2d(kernel_size=[1, 2]),
        nn.Dropout(p=0.3),

        # ResNetBlock(32, 32),
        # nn.MaxPool2d(kernel_size=1),

        ResNetBlock(32, 64),
        nn.MaxPool2d(kernel_size=[1, 2]),
        nn.Dropout(p=0.5),
    )
    return encoder


def baseline_decoder(n_classes, n_in=64, fc1_out=100):
    # fc1_out = 128  100, 32
    encoder = nn.Sequential(
        nn.Linear(n_in, fc1_out),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(fc1_out, n_classes)
    )
    return encoder


class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResNetBlock, self).__init__()

        self.conv_identity = conv_block(in_ch, out_ch, activation='relu', kernel_size=1, padding=0,
                                        stride=1)
        self.conv1 = conv_block(in_ch, out_ch, kernel_size=3, padding=1, stride=1)
        self.conv2 = conv_block(out_ch, out_ch, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)

        identity = self.conv_identity(identity)

        x += identity

        return x


class Resnet(nn.Module):
    def __init__(self, in_c, n_classes):
        super().__init__()

        self.encoder = res_encoder(in_c)

        self.decoder = baseline_decoder(n_classes, 64 * 3 * 2 * 2, 10)

    def forward(self, x):
        x = self.encoder(x)

        x = x.view(x.size(0), -1)  # flat

        x = self.decoder(x)

        return x


class ResSwin(nn.Module):
    def __init__(self, cfg_pth):
        super(ResSwin, self).__init__()
        self.encoder = res_encoder(1)
        self.cfg = Config.fromfile(cfg_pth)
        self.swin = build_model(self.cfg)

    def forward(self, x):
        x = self.encoder(x)
        x = self.swin(x)
        return x, 0


if __name__ == '__main__':
    import torch

    model = Resnet(1, 10)
    dummy = torch.zeros(4, 1, 128, 128 * 4)

    cfg_path = '../configs/swin_base_patch4_window7_224.yaml'
    model = ResSwin(cfg_path)
    out, _ = model(dummy)

    print(model(dummy).shape)
    print(model)
