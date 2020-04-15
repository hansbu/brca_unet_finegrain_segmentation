import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import PIL.Image as Image
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable
import os
import sys
import torch.nn as nn
import cv2
import random
import glob
import argparse
from skimage.color import hed2rgb, rgb2hed
import multiprocessing as mp


def load_imgs_files(data_path='data', limit=1000000, isTrain=True):
    train_imgs = []
    val_imgs = []

    train_img_path = os.path.join(data_path, 'TCGA_BRCA_finegrain_patches_10X')
    train_mask_path = os.path.join(data_path, 'TCGA_BRCA_finegrain_patches_10X_mask')
    train_img_fns = [os.path.join(train_img_path, f) for f in os.listdir(train_img_path) if len(f) > 3]

    val_img_path = os.path.join(data_path, 'TCGA_BRCA_finegrain_patches_10X_val')
    val_mask_path = os.path.join(data_path, 'TCGA_BRCA_finegrain_patches_10X_mask_val')
    val_img_fns = [os.path.join(val_img_path, f) for f in os.listdir(val_img_path) if len(f) > 3]

    random.shuffle(train_img_fns)
    val_img_fns.sort()
    val_paths = []

    c = 0
    if isTrain:
        for fn in train_img_fns:
            mask_fn = os.path.join(train_mask_path, fn.split('/')[-1].split('.png')[0] + '_mask.png')
            img = cv2.imread(fn)
            mask = cv2.imread(mask_fn, 0)

            mask = np.expand_dims(mask, axis=2)
            img_merged = np.concatenate((img, mask), axis=2)
            train_imgs.append(img_merged)
            c += 1
            if c % 10 == 0: print('Loading training data: {}/{}'.format(c, len(train_img_fns)))
            if c > limit: break

    c = 0
    for fn in val_img_fns:
        mask_fn = os.path.join(val_mask_path, fn.split('/')[-1].split('.png')[0] + '_mask.png')
        img = cv2.imread(fn)
        mask = cv2.imread(mask_fn, 0)

        mask = np.expand_dims(mask, axis=2)
        img_merged = np.concatenate((img, mask), axis=2)
        val_imgs.append(img_merged)
        val_paths.append(mask_fn.split('/')[-1])
        c += 1
        if c % 10 == 0: print('Loading val data: {}/{}'.format(c, len(val_img_fns)))
        if c > limit: break

    return train_imgs, val_imgs, val_paths


class data_loader(Dataset):
    """
    Dataset to read image and label for training
    """

    def __init__(self, imgs, transform=None, APS=224, isTrain=True):
        self.imgs = imgs
        self.transform = transform
        self.APS = APS
        self.randints = [i for i in range(0, self.imgs[0].shape[0] - APS + 1, 10)]
        self.len_rand = len(self.randints)
        self.isTrain = isTrain

    def __getitem__(self, index):
        data = self.imgs[index]
        x = self.randints[random.randint(0, self.len_rand - 1)]
        y = self.randints[random.randint(0, self.len_rand - 1)]
        data = data[x:x + self.APS, y:y + self.APS, :]

        # mirror and flip
        # if np.random.rand(1)[0] < 0.5 and self.isTrain:
        #     data = np.flip(data, 0);        # flip on axis 0, vertiaclly flip
        # if np.random.rand(1)[0] < 0.5 and self.isTrain:
        #     data = np.flip(data, 1);       # flip on axis 1, mirror

        img, mask = data[:, :, :3], data[:, :, 3]
        img = Image.fromarray(img.astype(np.uint8), 'RGB')
        img = self.transform(img)

        mask = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))  # torch.Tensor

        return img, mask

    def __len__(self):
        return len(self.imgs)


def parallelize_model(model):
    if torch.cuda.is_available():
        model = model.cuda()
        # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        cudnn.benchmark = True
    return model


def unparallelize_model(model):
    try:
        while 1:
            # to avoid nested dataparallel problem
            model = model.module
    except AttributeError:
        pass
    return model


def cvt_to_gpu(X):
    return Variable(X.cuda()) if torch.cuda.is_available() \
        else Variable(X)


def dice_coeff(preds, targets, no_class):
    dice = np.zeros(no_class)
    eps = 0.00001

    for c in range(no_class):
        Pr = preds == c
        Tr = targets == c

        if Pr.shape != Pr.shape:
            raise ValueError("Shape mismatch: Pr and Tr must have the same shape.")

        intersection = np.logical_and(Pr, Tr)
        dice[c] = (2.0 * intersection.sum() + eps) / (Pr.sum() + Tr.sum() + eps)

    return dice


def jaccard_coeff(preds, targets, no_class):
    jaccard = np.zeros(no_class)
    eps = 0.00001

    for c in range(no_class):
        Pr = preds == c
        Tr = targets == c

        if Pr.shape != Pr.shape:
            raise ValueError("Shape mismatch: Pr and Tr must have the same shape.")

        intersection = np.logical_and(Pr, Tr)
        union = np.logical_or(Pr, Tr)
        jaccard[c] = (intersection.sum() + eps) / (union.sum() + eps)

    return jaccard
