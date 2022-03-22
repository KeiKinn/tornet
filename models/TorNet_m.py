import torch
import torch.nn as nn
from models.CRNN_VAR import Transformer, SelfAttentionMultihead
from models.BCRes import TransitionBlock, BroadcastedBlock


class FConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, mp=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
            nn.Conv2d(out_ch, out_ch, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.mp = nn.MaxPool2d(mp)

    def forward(self, x):
        x = self.conv(x)
        x = self.mp(x)
        return x


class BCBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1, 1), mode='bc'):
        super().__init__()
        self.bc = nn.Sequential(
            TransitionBlock(in_ch, out_ch, S=4, stride=stride),
            BroadcastedBlock(out_ch, S=4)
        ) if mode == 'bc' else nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.do = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.bc(x)
        x = self.conv(x)
        x = self.do(x)
        return x


class IdBlock(nn.Module):
    def __init__(self, in_ch, out_ch, mp=(1, 1)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.mp = nn.MaxPool2d(mp)

    def forward(self, x):
        x = self.conv(x)
        x = self.mp(x)
        return x


class BCResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1, 1), mp=(1, 1), mode='bc', norm='in'):
        super().__init__()

        self.id = IdBlock(in_ch, out_ch, mp=mp)
        self.bc1 = BCBlock(in_ch, out_ch // 2, stride=stride, mode=mode)
        self.bc2 = BCBlock(out_ch // 2, out_ch, stride=stride, mode=mode)

        if norm == 'none':
            self.norm = nn.Identity()
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_ch)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(out_ch // 8, out_ch)

    def forward(self, x):
        identity = self.id(x)
        x = self.bc1(x)
        x = self.bc2(x)
        x = x + identity
        x = self.norm(x)
        return x


class PostNet(nn.Module):
    def __init__(self, in_ch, out_ch, num_classes, mode='fc'):
        super().__init__()
        self.mode = mode
        if self.mode == 'transformer':
            self.net = Transformer()
        elif self.mode == 'lstm_attn':
            self.net = nn.LSTM(512 * 5, 128, num_layers=1, batch_first=True, bidirectional=False)
            self.mh_attn = SelfAttentionMultihead(d_model=128, num_heads=4)
        elif self.mode == 'lstm':
            self.net = nn.LSTM(in_ch, out_ch, num_layers=1, batch_first=True, bidirectional=False)
        elif self.mode == 'attn':
            self.net = SelfAttentionMultihead(d_model=in_ch, num_heads=4)
            self.linear = nn.Linear(in_ch, out_ch)
        elif self.mode == 'fc':
            self.net = nn.Linear(in_ch, out_ch)

        self.post = nn.Sequential(
            nn.Linear(out_ch, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if self.mode == 'transformer':
            x = self.net(x)
            x = torch.mean(x, 0)
        elif self.mode == 'lstm_attn':
            x = self.net(x)
            x = self.mh_attn(x)
        elif self.mode == 'lstm':
            x, _ = self.net(x)
            x = x[:, -1, :]
        elif self.mode == 'attn':
            x = self.net(x)
            x = self.linear(x)
        elif self.mode == 'fc':
            x = torch.mean(x, dim=1)
            x = self.net(x)

        x = self.post(x)
        return x


class TorNet(nn.Module):
    def __init__(self, mode='bc', norm='in', post_mode='fc'):
        super().__init__()
        self.conv = FConv(1, 32, mp=2)

        self.bcres1 = BCResBlock(32, 128, stride=(2, 1), mp=(4, 1), mode=mode, norm=norm)
        self.bcres2 = BCResBlock(128, 512, mp=(1, 1), mode=mode, norm=norm)

        self.post = PostNet(512 * 16, 128, num_classes=2, mode=post_mode)

    def forward(self, x):
        x = self.conv(x)
        x = self.bcres1(x)
        x = self.bcres2(x)

        bs, ch, f, t = x.size()
        x = x.view(bs, ch * f, t).permute(0, 2, 1)  # bs, T, E

        x = self.post(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 1, 128, 800)
    model = TorNet(mode='bc', norm='in', post_mode='fc')
    y = model(x)
    print(model)
    print('num parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

