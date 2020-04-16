import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def eval_net(net, no_class, dataset, is_save=False):
    """Evaluation without the densecrf with the dice coefficient"""
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0")

    net.eval()
    tot = np.zeros(no_class)
    tot_jac = np.zeros(no_class)
    tot_loss = 0
    len_data = 1

    for i, data in enumerate(dataset):
        imgs, true_masks = data
        len_data += 1

        imgs = Variable(imgs.to(device))
        true_masks = Variable(true_masks.type(torch.LongTensor).to(device))

        masks_pred = net(imgs)
        loss = criterion(masks_pred, true_masks)
        tot_loss += loss.item()

        _, masks_pred = torch.max(masks_pred.data, 1)

        tot += dice_coeff(masks_pred.data.cpu().numpy(), true_masks.data.cpu().numpy(), no_class)      # tot is a numpy array
        tot_jac += jaccard_coeff(masks_pred.data.cpu().numpy(), true_masks.data.cpu().numpy(), no_class)      # tot is a numpy array

    return tot_loss/len_data, tot / len_data, sum(tot) / len(tot) / len_data, tot_jac / len_data, sum(tot_jac) / len(tot_jac) / len_data
