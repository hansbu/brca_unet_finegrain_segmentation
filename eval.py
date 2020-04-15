import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils import *

def eval_net(net, dataset, is_save=False):
    """Evaluation without the densecrf with the dice coefficient"""
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0")

    net.eval()
    no_class = net.n_classes
    tot = np.zeros(no_class)
    tot_jac = np.zeros(no_class)
    tot_loss = 0

    for i, data in enumerate(dataset):
        if i % 10 == 0: print('Processing {}/{}'.format(i, len(dataset)))
        imgs, true_masks = data

        imgs = Variable(imgs.to(device))
        true_masks = Variable(true_masks.type(torch.LongTensor).to(device))

        masks_pred = net(imgs)
        loss = criterion(masks_pred, true_masks)
        tot_loss += loss.item()

        _, masks_pred = torch.max(masks_pred.data, 1)

        tot += dice_coeff(masks_pred.data.cpu().numpy(), true_masks.data.cpu().numpy())      # tot is a numpy array
        tot_jac += jaccard_coeff(masks_pred.data.cpu().numpy(), true_masks.data.cpu().numpy())      # tot is a numpy array

    return tot_loss/(i + 1), tot / (i + 1), sum(tot) / len(tot) / (i + 1), tot_jac / (i + 1), sum(tot_jac) / len(tot_jac) / (i + 1)