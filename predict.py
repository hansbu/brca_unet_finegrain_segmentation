import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import copy
from torch.utils.data import DataLoader, Dataset
import pdb
from torchvision import transforms

from eval import eval_net
from unet import UNet
from utils import *
import cv2
from color_comp_predict import *
from train import get_data_transforms


def get_args():
    parser = OptionParser()
    parser.add_option('--model', '-m', default='CP584_resolution10_best_0.8450.pth', metavar='FILE',
                      help="Specify the file in which is stored the model (default : 'MODEL.pth')")
    parser.add_option('--N_limit', default=500000, type=int, help='limit the number of data to be loaded')
    parser.add_option('--num_workers', default=4, type=int, help='number of workers')
    parser.add_option('--APS', default=250, type=int, help='patch size of original input')

    (options, args) = parser.parse_args()
    return options


def load_model(no_class, model_file_name):
    if 'upLearned' in model_file_name:
        net = UNet(n_channels=3, n_classes=no_class, bilinear=False)
    else:
        net = UNet(n_channels=3, n_classes=no_class, bilinear=True)

    net = parallelize_model(net)
    net.load_state_dict(torch.load('checkpoints/' + model_file_name))
    print("Model loaded !")
    net.eval()
    return net


if __name__ == '__main__':
    args = get_args()

    out_fol_mask = 'data/predicted_masks/'
    out_fol_img = 'data/predicted_imgs/'
    if not os.path.exists(out_fol_mask): os.mkdir(out_fol_mask)
    if not os.path.exists(out_fol_img): os.mkdir(out_fol_img)

    no_class = 2
    resolution = 10
    data_transforms = get_data_transforms()

    print('================================Start loading data!')
    _, img_vals, val_paths = load_imgs_files(data_path='data', limit=args.N_limit, isTrain=False, resolution=resolution)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0")

    net = load_model(no_class, args.model)
    tot = np.zeros(no_class)
    tot_jac = np.zeros(no_class)
    APS = args.APS

    for i, data in enumerate(img_vals):
        if i % 10 == 0: print('Processing {}/{}'.format(i, len(img_vals)))

        patch_size = data.shape[1]      # data.shape 1000x1000x4
        num_splits = patch_size // APS
        imgs = torch.empty(num_splits*num_splits, 3, APS, APS)
        true_masks = torch.empty(num_splits*num_splits, APS, APS)
        predicted_mask = np.zeros((patch_size, patch_size))
        ind = 0
        for r in range(0, patch_size, APS):
            for c in range(0, patch_size, APS):
                tmp = data[r: r + APS, c: c + APS, :]
                img, true_mask = tmp[:, :, :3], tmp[:, :, 3]
                img = Image.fromarray(img.astype(np.uint8), 'RGB')
                imgs[ind] = data_transforms['val'](img)           # 3xAPSxAPS
                true_masks[ind] = torch.from_numpy(true_mask)     # APSxAPS
                ind += 1

        imgs = Variable(imgs.to(device))
        true_masks = Variable(true_masks.type(torch.LongTensor).to(device))

        masks_pred = net(imgs)
        _, masks_pred = torch.max(masks_pred.data, 1)

        masks_pred = masks_pred.data.cpu().numpy()
        true_masks = true_masks.data.cpu().numpy()
        tot += dice_coeff(masks_pred, true_masks, no_class)               # tot is a numpy array
        tot_jac += jaccard_coeff(masks_pred, true_masks, no_class)        # tot is a numpy array

        ind = 0
        for r in range(0, patch_size, APS):
            for c in range(0, patch_size, APS):
                predicted_mask[r: r + APS, c: c + APS] = masks_pred[ind]
                ind += 1

        cv2.imwrite(os.path.join(out_fol_mask, val_paths[i]), predicted_mask)

    color_comp_main('data/TCGA_BRCA_finegrain_patches_{}X_val'.format(resolution), out_fol_mask, out_fol_img)
    print('Prediction:\t Dice: {} \t Averaged Dice: {:.4f} \t Jacc: {} \t Averaged Jacc: {:.4f}'.format(
                tot / (i + 1), sum(tot) / len(tot) / (i + 1), tot_jac / (i + 1), sum(tot_jac) / len(tot_jac) / (i + 1)))