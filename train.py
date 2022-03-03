import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from mmcv import Config
from models import build_model
from models.model import SwT
from train_cfg import TrainConfig
from DataLoader.DCASE_PPL import DCASE_MFCC
import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def get_path_prefix():
    machine = os.uname()[1]
    prefix = ''
    local = False
    if machine == 'eihw253':
        prefix = '/home/xinjing/Documents/gpu5'
        local = True
    return prefix, local


def save_model(save_path, current_model, current_epoch, marker, timestamp):
    save_path = os.path.join(save_path, timestamp)
    save_to = os.path.join(save_path, '{}_{}.pkl'.format(marker, current_epoch))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(current_model, save_to)
    print('<== Model is saved to {}'.format(save_to))


def print_nn(mm):
    def count_pars(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    num_pars = count_pars(mm)
    print(mm)
    print('# pars: {}'.format(num_pars))
    print('{} : {}'.format('device', device))


def print_flags(cfg):
    print('--------------------------- Flags -----------------------------')
    for flag in cfg.asdic():
        print('{} : {}'.format(flag, getattr(cfg, flag)))


def report_metrics(pred_aggregate_, gold_aggregate_):
    assert len(pred_aggregate_) == len(gold_aggregate_)
    print('# samples: {}'.format(len(gold_aggregate_)))
    print(classification_report(gold_aggregate_, pred_aggregate_))
    print(confusion_matrix(gold_aggregate_, pred_aggregate_))


def create_ds(cfg):
    ds_tr = DCASE_MFCC(cfg.tr_sd, 16000, 115200)
    ds_ev = DCASE_MFCC(cfg.val_sd, 16000, 115200)
    return ds_tr, ds_ev


def create_tr_dl(dataset, batch_size=4):
    dl_tr = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dl_tr


def create_val_dl(dataset, batch_size=4):
    val_dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return val_dl


def create_dl(cfg):
    ds_tr, ds_val = create_ds(cfg)
    dl_tr = create_tr_dl(ds_tr, cfg.tr_bs)
    dl_val = create_val_dl(ds_val, cfg.val_bs)
    return dl_tr, dl_val


def training_setting(model, lr=1e-4):
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()
    return optimizer, loss_fn


def train(dl, optimizer, loss_fn, epoch, log_freq=10):
    losses = 0.
    counter = 1
    tmp_losses = 0.
    tmp_counter = 0

    correct = 0
    total = 0
    tmp_correct = 0
    tmp_total = 0

    for idx, (batch, label) in enumerate(dl):
        batch = batch.to(device)
        label = label.to(device)

        out = model(batch)
        del batch
        optimizer.zero_grad()
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()

        losses += loss.item()
        counter += 1
        correct += sum(np.argmax(out.detach().cpu().numpy(), axis=1) == label.detach().cpu().numpy())
        total += len(label.detach().cpu().numpy())

        tmp_losses += loss.item()
        tmp_counter += 1
        tmp_correct += sum(np.argmax(out.detach().cpu().numpy(), axis=1) == label.detach().cpu().numpy())
        tmp_total += len(label.detach().cpu().numpy())

        if idx % log_freq == 0:
            print('  [{}][{}] loss: {:.4f}, Acc: {:.4}'.format(epoch, idx,
                                                               tmp_losses / tmp_counter,
                                                               tmp_correct / tmp_total))

            tmp_losses = 0.
            tmp_counter = 0
            tmp_correct = 0
            tmp_total = 0

    print('##> [{}] Train loss: {:.4f}, Acc: {:.4}'.format(epoch, losses / counter, correct / total))


def eval(dl, loss_fn):
    losses = 0.
    counter = 0
    correct = 0
    total = 0
    pred_aggregate = []
    gold_aggregate = []
    model.eval()

    for idx, (batch, label) in enumerate(dl):
        with torch.no_grad():
            batch = batch.to(device)
            label = label.to(device)

            out = model(batch)
        del batch
        loss = loss_fn(out, label)
        losses += loss.item()
        counter += 1
        correct += sum(np.argmax(out.detach().cpu().numpy(), axis=1) == label.detach().cpu().numpy())
        total += len(label.detach().cpu().numpy())

        pred_aggregate.extend(np.argmax(out.detach().cpu().numpy(), axis=1).tolist())
        gold_aggregate.extend(label.detach().cpu().numpy().tolist())

    print('==> Val loss: {:.4f}, Acc: {:.5}'.format(losses / counter, correct / total))
    report_metrics(pred_aggregate, gold_aggregate)
    print('-' * 80)


if __name__ == "__main__":
    machine, _ = get_path_prefix()
    save_path = machine + '/nas/staff/data_work/Sure/1_Xin/Swin/models/'
    cfg_path = './configs/swin_base_patch4_window7_224.yaml'
    tr_cfg = TrainConfig()
    print_flags(tr_cfg)
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # md_cfg = Config.fromfile(cfg_path)
    # model = build_model(md_cfg)
    model = SwT(cfg_path)
    model.to(device)
    print_nn(model)
    print('-' * 80)

    tr_dl, val_dl = create_dl(tr_cfg)
    optimizer, loss_fn = training_setting(model, tr_cfg.lr)

    for epoch in range(1, tr_cfg.epoches + 1):
        train(tr_dl, optimizer, loss_fn, epoch, tr_cfg.log_freq)
        if epoch % tr_cfg.save_every == 0:
            save_model(save_path, model, epoch, tr_cfg.md_name, timestamp)

        if epoch % tr_cfg.val_every == 0:
            eval(val_dl, loss_fn)
