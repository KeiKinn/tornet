import os
import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import Resample
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, ComputeDeltas, TimeMasking, FrequencyMasking

from torch.utils.data import Dataset, DataLoader
import pickle as p
import random

from utils.spec import SpecAugmentation

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed_path', type=str, default='/nas/staff/data_work/Sure/AudioSet2021/all_seeds.pkl')
parser.add_argument('--bs', type=int, default=128)
args = parser.parse_args()

def get_path_prefix():
    machine = os.uname()[1]
    prefix = ''
    # Local machine
    if '253' in machine:
        prefix = '/home/xinjing/Documents/gpu5'
    return prefix

class AudioSet(Dataset):
    def __init__(self, seeds_path, uni_sr, segment_len, sample_segments=False, num_ref_segments=None):
        super(AudioSet, self).__init__()
        with open(seeds_path, 'rb') as seeds_file:
            self.seeds = p.load(seeds_file)

        self.uni_sr = uni_sr
        self.sample_segments = sample_segments
        if sample_segments:
            self.num_ref_segments = num_ref_segments if num_ref_segments is not None else 3
        self.segment_len = segment_len

    def __getitem__(self, idx):
        machine = get_path_prefix()
        seed_path = machine+self.seeds[idx]
        wavform, sr = torchaudio.load(seed_path, normalize=True)
        wavform = self.unify_data(wavform, sr, self.uni_sr)
        if self.sample_segments:
            tgt_segment, neg_segments = self.truncate_segments(wavform)
            return tgt_segment, neg_segments
        else:
            tgt_segment = self.truncate_segments(wavform)
            return tgt_segment

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
        """

        :param wf:
        :param num_neg_segs:
        :param sample: "self" => permute the frames of the target segment,
                       'global' => sample from all frames,
                       'segments' => new segments
        :return:
        """
        wf_len = wf.size()[1]
        if wf_len < self.segment_len:
            repeat_time = int(self.segment_len / wf_len) + 1
            wf = torch.cat([wf] * repeat_time, dim=1)
            wf_len = wf.size()[1]
        ss_latest = wf_len - self.segment_len
        ss_available = list(range(ss_latest + 1))
        ss = random.choice(ss_available)
        tgt_segment = wf[:, ss:ss + self.segment_len]
        if self.sample_segments:
            ss_available.remove(ss)
            wf = torch.cat([wf, wf[:, :self.segment_len - 1]], dim=1)
            ref_segments = [wf[:, ss: ss + self.segment_len]
                            for ss in random.sample(ss_available, self.num_ref_segments)]
            ref_segments = torch.stack(ref_segments, dim=0)
            return tgt_segment, ref_segments
        return tgt_segment


class AudioSetMFCC(AudioSet):
    def __init__(self, seeds_path, uni_sr, segment_len, sample_segments=False, num_ref_segments=None):
        super(AudioSetMFCC, self).__init__(seeds_path, uni_sr, segment_len, sample_segments, num_ref_segments)
        self.mel_spec_extractor = nn.Sequential(MelSpectrogram(uni_sr, n_fft=640, n_mels=128), AmplitudeToDB())
        self.mel_spec_delta = nn.Sequential(ComputeDeltas())
        self.mask = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                     freq_drop_width=8, freq_stripes_num=2)

    def __getitem__(self, idx):
        machine = get_path_prefix()
        seed_path = machine+self.seeds[idx]
        wavform, sr = torchaudio.load(seed_path, normalize=True)
        wavform = self.unify_data(wavform, sr, self.uni_sr)
        tgt_segment = self.truncate_segments(wavform)
        tgt_segment = self.compute_output(tgt_segment)
        ref_segments = self.mask(tgt_segment)
        return tgt_segment, ref_segments

    def compute_output(self, data, dim=0):
        data = self.mel_spec_extractor(data)
        # data_delta = self.mel_spec_delta(data)
        # data_delta2 = self.mel_spec_delta(data_delta)
        # data = torch.cat([data, data_delta, data_delta2], dim=dim)
        return data


if __name__ == '__main__':
    SAMPLE_SEGMENTS = True
    machine = get_path_prefix()
    tr = machine + '/nas/staff/data_work/Sure/librispeech/train.pkl'
    ds_tr = AudioSetMFCC(tr, 16000, segment_len=16000*5)
    dl = DataLoader(ds_tr, batch_size=2, shuffle=False, num_workers=1)
    for idx, batch in enumerate(dl):
        print(idx)

