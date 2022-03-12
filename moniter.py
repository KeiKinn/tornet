import os
from os import uname
from os.path import join
from moniter_tr import monitor_tr, monitor_uar

if 'eihw-gpu' in uname()[1]:
    root = '/home/liushuo/'
else:
    root = '/home/xinjing/Documents/gpu5/home/liushuo/'

slurm_id = '8910_0'

slurm = join(root, 'xin/swin_dev/slurm', slurm_id)

slurm_digit = int(slurm_id.split('_')[1])

if slurm_digit == 0:
    monitor_uar(slurm)

