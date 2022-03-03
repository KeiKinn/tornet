import os
import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import Resample
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, ComputeDeltas
from torch.utils.data import Dataset, DataLoader
import pickle as p
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed_path', type=str,
                    default='/nas/staff/data_work/Sure/DCASE2021/processed_data/seeds/test.pkl')
parser.add_argument('--bs', type=int, default=10)
args = parser.parse_args()


def get_path_prefix():
    machine = os.uname()[1]
    prefix = ''
    # Local machine
    if '253' in machine:
        prefix = '/home/xinjing/Documents/gpu5'
    return prefix


class DCASE_PPL(Dataset):
    def __init__(self, seed_path, uni_sr, segment_len, spectrum_mode=False):
        super(DCASE_PPL, self).__init__()
        with open(seed_path, 'rb') as seed_file:
            self.seeds = p.load(seed_file)

        self.uni_sr = uni_sr
        self.segment_len = segment_len
        self.label2num = {
            'airport': 0,
            'shopping_mall': 1,
            'metro_station': 2,
            'street_pedestrian': 3,
            'public_square': 4,
            'street_traffic': 5,
            'tram': 6,
            'bus': 7,
            'metro': 8,
            'park': 9
        }

        self.spectrum_mode = spectrum_mode
        self.mel_spec_extractor = nn.Sequential(MelSpectrogram(uni_sr, n_fft=640, n_mels=128), AmplitudeToDB())
        self.labels = {}
        for label in self.label2num:
            self.labels[label] = [idx for idx, (_, _, lbl) in enumerate(self.seeds) if lbl == label]

    def __getitem__(self, idx):
        machine = get_path_prefix()
        seed = self.seeds[idx]
        audio_path, label = machine + seed[0], self.label2num[seed[2]]
        wavform, sr = torchaudio.load(audio_path, normalize=True)
        wavform = self.unify_data(wavform, sr, self.uni_sr)
        if self.spectrum_mode:
            mel_spec = self.mel_spec_extractor(wavform)
            tgt_segment = mel_spec[:, :, :49]
        else:
            tgt_segment = self.truncate_segments(wavform)
        return tgt_segment, label

    def __len__(self):
        return len(self.seeds)

    def unify_data(self, wf, fs, uni_fs):
        ch = wf.size()[0]
        if ch > 1:
            wf = torch.mean(wf, dim=0, keepdim=True)

        if fs != uni_fs:
            resampler = Resample(fs, uni_fs)
            wf = resampler(wf)
        return wf

    def truncate_segments(self, wf):
        # wf_len = wf.size()[1]
        # ss_latest = wf_len - self.segment_len
        # ss_available = list(range(ss_latest + 1))
        # ss = random.choice(ss_available)
        ss = 0
        tgt_segment = wf[:, ss:ss + self.segment_len]
        # if sample == 'self':
        #     neg_segments = self.permute_frames(tgt_segment, self.num_neg_segments)
        # elif sample == 'global':
        #     permuted_frames = self.permute_frames(wf)
        return tgt_segment


class DCASE_MFCC(DCASE_PPL):
    def __init__(self, seed_path, uni_sr, segment_len, spectrum_mode=False):
        super(DCASE_MFCC, self).__init__(seed_path, uni_sr, segment_len, spectrum_mode)
        self.mel_spec_extractor = nn.Sequential(MelSpectrogram(uni_sr, n_fft=1024, n_mels=224), AmplitudeToDB())
        self.mel_spec_delta = nn.Sequential(ComputeDeltas())

    def __getitem__(self, idx):
        machine = get_path_prefix()
        seed = self.seeds[idx]
        audio_path, label = machine + seed[0], self.label2num[seed[2]]
        wavform, sr = torchaudio.load(audio_path, normalize=True)
        wavform = self.unify_data(wavform, sr, self.uni_sr)
        tgt_segment = self.truncate_segments(wavform)
        tgt_segment = self.compute_output(tgt_segment)
        return tgt_segment, label

    def compute_output(self, data):
        data = self.mel_spec_extractor(data)[:, :, :224]
        # data_delta = self.mel_spec_delta(data)
        # data_delta2 = self.mel_spec_delta(data_delta)
        # data = torch.cat([data, data_delta, data_delta2], dim=0)
        return data


if __name__ == '__main__':
    ds = DCASE_PPL(args.seed_path, 16000, segment_len=16000, spectrum_mode=True)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=1)
    for idx, (batch, label) in enumerate(dl):
        print(idx, batch.size(), label)
        if idx == 5:
            break
