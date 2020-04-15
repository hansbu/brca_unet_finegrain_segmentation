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


def get_args():
    parser = OptionParser()
    parser.add_option('--model', '-m', default='CP544.pth',
                        metavar='FILE', help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_option('--N_limit', default=500000, type=int,
                      help='limit the number of data to be loaded')
    parser.add_option('--num_workers', default=4,
                      type=int, help='number of workers')
    parser.add_option('--APS', default=400, type=int,
                      help='patch size of original input')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    out_fol_mask = 'predicted_masks/'
    out_fol_img = 'predicted_imgs/'

    if not os.path.exists(out_fol_mask): os.mkdir(out_fol_mask)
    if not os.path.exists(out_fol_img): os.mkdir(out_fol_img)

    mean = [0.6462,  0.5070,  0.8055]      # for Prostate cancer
    std = [0.1381,  0.1674,  0.1358]

    data_transforms = {
        'train': transforms.Compose([           # 2 steps of data augmentation for training
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]),

        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    }

    print('================================Start loading data!')
    _, img_vals, val_paths = load_imgs_files(data_path='data', limit=args.N_limit, isTrain=False)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0")

    no_class = 4
    net = UNet(n_channels=3, n_classes=no_class)
    net = parallelize_model(net)
    net.load_state_dict(torch.load('checkpoints/' + args.model))
    print("Model loaded !")
    net.eval()

    tot = np.zeros(no_class)
    tot_jac = np.zeros(no_class)

    for i, data in enumerate(img_vals):
        if i % 10 == 0: print('Processing {}/{}'.format(i, len(img_vals)))

        imgs = torch.empty(9, 3, 400, 400)
        true_masks = torch.empty(9, 400, 400)
        predicted_mask = np.zeros((1200, 1200))
        ind = 0
        for r in [0, 400, 800]:
            for c in [0, 400, 800]:
                tmp = data[r: r + 400, c: c + 400, :]
                img, true_mask = tmp[:, :, :3], tmp[:, :, 3]
                img = Image.fromarray(img.astype(np.uint8), 'RGB')
                imgs[ind] = data_transforms['val'](img)           # 3x400x400
                true_masks[ind] = torch.from_numpy(true_mask)     # 400x400
                ind += 1

        imgs = Variable(imgs.to(device))
        true_masks = Variable(true_masks.type(torch.LongTensor).to(device))

        masks_pred = net(imgs)
        _, masks_pred = torch.max(masks_pred.data, 1)

        masks_pred = masks_pred.data.cpu().numpy()
        true_masks = true_masks.data.cpu().numpy()
        tot += dice_coeff(masks_pred, true_masks)               # tot is a numpy array
        tot_jac += jaccard_coeff(masks_pred, true_masks)        # tot is a numpy array

        ind = 0
        for r in [0, 400, 800]:
            for c in [0, 400, 800]:
                predicted_mask[r: r + 400, c: c + 400] = masks_pred[ind]
                ind += 1

        predicted_mask[predicted_mask == 3] = 4
        predicted_mask[predicted_mask == 2] = 3

        cv2.imwrite(os.path.join(out_fol_mask, val_paths[i]), predicted_mask)

    color_comp_main('data/val_jpg_ext', out_fol_mask, out_fol_img)

    print('Prediction:\t Dice: {} \t Averaged Dice: {:.4f} \t Jacc: {} \t Averaged Jacc: {:.4f}'.format(
    tot / (i + 1), sum(tot) / len(tot) / (i + 1), tot_jac / (i + 1), sum(tot_jac) / len(tot_jac) / (i + 1)))