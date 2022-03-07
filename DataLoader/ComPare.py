import os
import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, Resample
import csv
import torch.utils.data as data
import random


def get_path_prefix():
    machine = os.uname()[1]
    prefix = ''
    # Local machine
    if '253' in machine:
        prefix = '/home/xinjing/Documents/gpu5'
    return prefix


def get_data_path(key):
    compare = {
        'train': '/nas/staff/data_work/Sure/ComParE21/lab/train.csv',
        'test': '/nas/staff/data_work/Sure/ComParE21/lab/test.csv',
        'val': '/nas/staff/data_work/Sure/ComParE21/lab/devel.csv'
    }

    return compare[key]


class Compare(data.Dataset):
    def __init__(self, dataset, seg_len, uni_sr=16000):
        machine = get_path_prefix()
        root = '/nas/staff/data_work/Sure/ComParE21/wav/'
        self.uni_sr = uni_sr
        self.segment_len = seg_len
        self.root = machine + root
        self.path = machine + get_data_path(dataset)
        self.rows = self.get_data()

        self.mel_spec_extractor = nn.Sequential(MelSpectrogram(uni_sr, n_fft=1024, hop_length=256, n_mels=128),
                                                AmplitudeToDB())
        self.label_dict = {
            'negative': 0,
            'positive': 1
        }

    def __getitem__(self, index):
        row = self.rows[index]
        wav_path = self.root + row[0]
        label = self.label_dict[row[1]]
        wav_data, sr = torchaudio.load(wav_path, normalize=True)
        wav_data = self.unify_data(wav_data, sr, self.uni_sr)
        tgt_segment = self.truncate_segments(wav_data)
        tgt_segment = self.compute_output(tgt_segment)
        return tgt_segment, label

    def __len__(self):
        return len(self.rows)

    def get_data(self):
        rows = []
        with open(self.path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)
            rows.pop(0)
        return rows

    def unify_data(self, wf, fs, uni_fs):
        ch = wf.size()[0]
        if ch > 1:
            wf = torch.mean(wf, dim=0, keepdim=True)

        if fs != uni_fs:
            resampler = Resample(fs, uni_fs)
            wf = resampler(wf)
        return wf

    def truncate_segments(self, wf):
        wf_len = wf.size()[1]
        if wf_len < self.segment_len:
            repeat_time = int(self.segment_len / wf_len) + 1
            wf = torch.cat([wf] * repeat_time, dim=1)
            wf_len = wf.size()[1]

        ss_latest = wf_len - self.segment_len
        ss_available = list(range(ss_latest + 1))
        ss = random.choice(ss_available)
        tgt_segment = wf[:, ss:ss + self.segment_len]
        return tgt_segment

    def compute_output(self, data):
        data = self.mel_spec_extractor(data)[:, :, :512]
        return data


if __name__ == '__main__':
    compare = Compare(dataset='train', seg_len=16000 * 3, uni_sr=16000)
    dl = data.DataLoader(compare, batch_size=2, shuffle=False)
    for i, (data, label) in enumerate(dl):
        print(data.size())
