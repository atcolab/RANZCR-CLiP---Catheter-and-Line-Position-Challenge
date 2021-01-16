import os
import sys
import time
import math
import numpy as np
import pandas as pd
from shutil import copyfile
import matplotlib.pyplot as plt

from model import resnet34
import config
import engine
import dataset
import transforms

import warnings
warnings.filterwarnings("ignore")

import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensor

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from apex import amp
import timm
import pretrainedmodels
from torchcontrib.optim import SWA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CyclicLR
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from torch.utils.data import DataLoader, Dataset

def run():

    df = pd.read_csv('../input/train.csv')
    target_cols = df.iloc[:, 1:12].columns.tolist()

    folds = df.copy()
    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(folds)):

        train_test = folds.iloc[train_idx]
        train_test.reset_index(drop=True, inplace=True)  

        valid_test = folds.iloc[valid_idx]
        valid_test.reset_index(drop=True, inplace=True)

        train_dataset = dataset.RANZCRDataset(
            train_test,
            train_test[target_cols],
            transforms.transforms_train
        )

        valid_dataset = dataset.RANZCRDataset(
            valid_test,
            valid_test[target_cols],
            transforms.transforms_valid
        )

        train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BS, num_workers=config.NUM_WORKERS, sampler=RandomSampler(train_dataset))
        valid_loader = DataLoader(valid_dataset, batch_size=config.VALID_BS, num_workers=config.NUM_WORKERS, sampler=SequentialSampler(valid_dataset))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = resnet34().to(device)
        optimizer = Adam(model.parameters(), lr=config.LR)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        scheduler = CosineAnnealingLR(optimizer, config.NUM_EPOCHS)
        loss_func = nn.BCEWithLogitsLoss().to(device)

        best_file = f'{config.KERNEL_TYPE}_best_fold{fold}.bin'
        roc_auc_max = 0

        for epoch in range(config.NUM_EPOCHS):

            scheduler.step(epoch)
            avg_train_loss = engine.train_loop_fn(model, train_loader, optimizer, loss_func, device)
            avg_val_loss, PREDS, TARGS = engine.val_loop_fn(model, valid_loader, optimizer, loss_func, device)

            roc_auc = roc_auc_score(PREDS, TARGS, average='macro')
            print(f"Epoch: {epoch+1} | lr: {optimizer.param_groups[0]['lr']:.7f} | train loss: {avg_train_loss:.4f} | val loss: {avg_val_loss:.4f} | roc auc score: {roc_auc:.4f}")

            if roc_auc > roc_auc_max:
                print('score2 ({:.6f} --> {:.6f}).  Saving model ...'.format(roc_auc_max, roc_auc))
                torch.save(model.state_dict(), best_file)
                roc_auc_max = roc_auc

            torch.save(model.state_dict(), f'{config.KERNEL_TYPE}_final_fold.bin')

if __name__ == "__main__":
    run()