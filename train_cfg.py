import os
from dataclasses import dataclass, asdict


def get_path_prefix():
    machine = os.uname()[1]
    prefix = ''
    local = False
    if machine == 'eihw253':
        prefix = '/home/xinjing/Documents/gpu5'
        local = True
    return prefix, local


def get_dcase_path(year):
    path = {
        '2021': ['/nas/staff/data_work/Sure/DCASE2021/processed_data/seeds/train.pkl',
                 '/nas/staff/data_work/Sure/DCASE2021/processed_data/seeds/test.pkl'],
        '2019': ['/nas/staff/data_work/Sure/DCASE2019/train_label.pkl',
                 '/nas/staff/data_work/Sure/DCASE2019/test_label.pkl'],
    }
    return path[year]


@dataclass
class TrainConfig:
    machine, local = get_path_prefix()
    sd = get_dcase_path('2021')
    tr_sd: str = machine + sd[0]
    val_sd: str = machine + sd[1]
    md_name: str = 'Swin_T'
    lr: float = 1e-5
    tr_bs: int = 2 if local else 64
    val_bs: int = 2 if local else 64
    tr_sm: int = 20 if local else 20000
    va_sm: int = 20 if local else 20000
    save_every: int = 10000 if local else 2000
    val_every: int = 1
    epoches: int = 100
    log_freq: int = 20

    def asdic(self):
        return asdict(self)
