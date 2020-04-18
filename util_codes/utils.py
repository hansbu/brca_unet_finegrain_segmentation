import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import PIL.Image as Image
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import os
import cv2
import random

random.seed(12345)


def load_imgs_paths(img_fns, train_mask_path, limit):
    out = []
    paths = []
    for c, fn in enumerate(img_fns):
        mask_fn = mask_path_from_img_path(train_mask_path, fn)
        img_merged = load_img_mask(fn, mask_fn)
        out.append(img_merged)
        paths.append(mask_fn.split('/')[-1])
        if c % 10 == 0: print('Loading data: {}/{}'.format(c, len(img_fns)))
        if c > limit: break
    return out, paths


def mask_path_from_img_path(mask_fol, img_path):
    return os.path.join(mask_fol, img_path.split('/')[-1].split('.png')[0] + '_mask.png')


def load_img_mask(img_path, mask_path):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)

    mask = np.expand_dims(mask, axis=2)
    img_merged = np.concatenate((img, mask), axis=2)
    return img_merged


def load_imgs_files(data_path='data', limit=1000000, isTrain=True, resolution=10):
    train_imgs, val_imgs = [], []

    train_img_path = os.path.join(data_path, 'TCGA_BRCA_finegrain_patches_{}X'.format(resolution))
    train_mask_path = os.path.join(data_path, 'TCGA_BRCA_finegrain_patches_{}X_mask'.format(resolution))
    train_img_fns = [os.path.join(train_img_path, f) for f in os.listdir(train_img_path) if len(f) > 3]

    test_img_path = os.path.join(data_path, 'TCGA_BRCA_finegrain_patches_{}X_val'.format(resolution))
    test_mask_path = os.path.join(data_path, 'TCGA_BRCA_finegrain_patches_{}X_mask_val'.format(resolution))
    test_img_fns = [os.path.join(test_img_path, f) for f in os.listdir(test_img_path) if len(f) > 3]

    train_img_fns.sort()
    random.shuffle(train_img_fns)
    test_img_fns.sort()

    if isTrain:
        val_img_fns = train_img_fns[:50]
        train_img_fns = train_img_fns[50:]
        train_imgs, _ = load_imgs_paths(train_img_fns, train_mask_path, limit)
        val_imgs, _ = load_imgs_paths(val_img_fns, train_mask_path, limit)

    test_imgs, test_paths = load_imgs_paths(test_img_fns, test_mask_path, limit)

    return train_imgs, val_imgs, test_imgs, test_paths


def get_augment(img, APS):
    out = []
    img_size = img.shape[0]
    for r in range(0, img_size, APS//2):
        for c in range(0, img_size, APS // 2):
            if r + APS < img_size and c + APS < img_size:
                tmp = img[r:r + APS, c:c + APS, :]
                out.append(tmp)
    return out


def augment_val(imgs, APS):
    out = []
    for img in imgs:
        out.extend(get_augment(img, APS))
    return out


class data_loader(Dataset):
    """
    Dataset to read image and label for training
    """

    def __init__(self, imgs, transform=None, APS=224, isTrain=True):
        self.imgs = imgs
        self.transform = transform
        self.APS = APS
        self.randints = [i for i in range(0, self.imgs[0].shape[0] - APS, 25)]
        self.len_rand = len(self.randints)
        self.isTrain = isTrain
        if not self.isTrain:
            self.imgs = augment_val(self.imgs, self.APS)

    def __getitem__(self, index):
        data = self.imgs[index]
        if self.isTrain:
            x = self.randints[random.randint(0, self.len_rand - 1)]
            y = self.randints[random.randint(0, self.len_rand - 1)]
            data = data[x:x + self.APS, y:y + self.APS, :]
        else:
            # this is validation
            offset = (data.shape[0] - self.APS) // 2
            data = data[offset:offset + self.APS, offset:offset + self.APS, :]  # get the central patch

        # mirror and flip
        if np.random.rand(1)[0] < 0.5 and self.isTrain:
            data = np.flip(data, 0)         # flip on axis 0, vertiaclly flip
        if np.random.rand(1)[0] < 0.5 and self.isTrain:
            data = np.flip(data, 1)         # flip on axis 1, mirror

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
