from os import uname, path
from os.path import join
import matplotlib.pyplot as plt

if 'rz.uni-augsburg.de' in uname()[1]:
    root = '/User/jx/PycharmProject/gpu5'
else:
    root = '/home/xinjing/Documents/gpu5/home/liushuo/'

filepath = path.abspath(__file__)
slurm = join(root, '../../slurm', '8317_0')
pass


def monitor_tr(slurm):
    tr_losses = []
    tr_acc = []

    va_losses = []
    va_acc = []

    with open(slurm) as slurm_file:
        lines = slurm_file.readlines()
        for line in lines:
            if line.startswith('##>'):
                elements = line.split(' ')
                if elements[2] == 'Train':
                    loss = float(elements[4].replace(',', ''))
                    tr_losses.append(loss)
                    acc = float(elements[6].replace('\n', ''))
                    tr_acc.append(acc)
            elif line.startswith('==>'):
                elements = line.split(' ')
                if elements[1] == 'Val':
                    loss = float(elements[3].replace(',', ''))
                    acc = float(elements[5].replace('\n', ''))
                    va_losses.append(loss)
                    va_acc.append(acc)
    print('max tr acc:', max(tr_acc))
    print('max va acc:', max(va_acc))
    plt.figure()
    plt.subplot(121)
    plt.plot(tr_losses)
    plt.plot(va_losses)
    plt.title('loss')
    plt.legend(['tr_loss', 'va_loss'])
    plt.subplot(122)
    plt.plot(tr_acc)
    plt.plot(va_acc)
    plt.title('acc')
    plt.legend(['tr_acc', 'va_acc'])
    plt.show()


def monitor_uar(slurm):
    uars = []

    with open(slurm) as slurm_file:
        lines = slurm_file.readlines()
        for line in lines:
            if line.startswith('#=>'):
                elements = line.split(' ')
                if elements[1] == 'Val':
                    uar = float(elements[3].replace('\n', ''))
                    uars.append(uar)

    print('max uar:', max(uars))
    plt.figure()
    plt.plot(uars)
    plt.title('uar')
    plt.show()
