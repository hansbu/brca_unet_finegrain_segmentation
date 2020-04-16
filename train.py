import sys
from optparse import OptionParser
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import time

from eval import eval_net
from unet import UNet
from utils import *


def train_net(net, train_loader=None, val_loader=None, args=None):

    dir_checkpoint = 'checkpoints/'
    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)

    best_dice = 0
    device = torch.device("cuda:0")
    print('Start Training: ', args)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        if epoch < 40:
            lr = args.lr
        elif epoch < 80:
            lr = args.lr / 2
        elif epoch < 100:
            lr = args.lr / 10
        elif epoch < 150:
            lr = args.lr / 50
        else:
            lr = args.lr / 100

        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        net.train()
        start = time.time()

        epoch_loss = 0
        train_dice = np.zeros(args.n_classes)
        train_jacc = np.zeros(args.n_classes)

        for i, data in enumerate(train_loader):
            imgs, true_masks = data

            imgs = Variable(imgs.to(device))
            true_masks = Variable(true_masks.type(torch.LongTensor).to(device))

            masks_pred = net(imgs)
            # print(imgs.size(), true_masks.size(), masks_pred.size())		# (bs, 3, 224, 224), [bs, 224, 224], [bs, 4, 224, 224]

            loss = criterion(masks_pred, true_masks)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, masks_pred = torch.max(masks_pred.data, 1)
            train_dice += dice_coeff(masks_pred.data.cpu().numpy(), true_masks.data.cpu().numpy(), args.n_classes)
            train_jacc += jaccard_coeff(masks_pred.data.cpu().numpy(), true_masks.data.cpu().numpy(), args.n_classes)

        print('Train Epoch: {} \t Train_Loss: {:.4f} \t Dice: {} \t Averaged Dice: {:.4f} \t Jacc: {} \t '
              'Averaged Jacc: {:.4f} \t Time: {:.2f} mins'.format(epoch, epoch_loss / (i + 1),
                                                                  train_dice / (i + 1),
                                                                  sum(train_dice) / len(train_dice) / (i + 1),
                                                                  train_jacc / (i + 1),
                                                                  sum(train_jacc) / len(train_jacc) / (i + 1),
                                                                  (time.time() - start) / 60.0))

        start = time.time()
        if (epoch + 1) % 2 == 0:  # perform evaluation
            val_loss, val_dice, averaged_dice, val_jacc, averaged_jacc = eval_net(net, args.n_classes, val_loader)
            print(
                'Validation Epoch: {} \t Val_Loss: {:.4f} \t Dice: {} \t Averaged Dice: {:.4f} \t Jacc: {} \t Averaged '
                'Jacc: {:.4f} \t Time: {:.2f} mins'.format(epoch + 1, val_loss, val_dice, averaged_dice, val_jacc,
                                                           averaged_jacc, (time.time() - start) / 60.0))

            if (epoch + 1) > 50 and best_dice < averaged_dice:
                best_dice = averaged_dice
                torch.save(net.state_dict(), dir_checkpoint + 'CP{}_resolution{}_upLearned_best_{}.pth'.format(
                                                                epoch + 1, args.resolution, best_dice))


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=1000, type='int', help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=32, type='int', help='batch size')
    parser.add_option('-l', '--lr', '--learning-rate', dest='lr', default=0.01, type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load', default=False, help='load file model')
    parser.add_option('--N_limit', default=500000, type=int, help='limit the number of data to be loaded')
    parser.add_option('--num_workers', default=8, type=int, help='number of workers')
    parser.add_option('--APS', default=224, type=int, help='patch size of original input')
    parser.add_option('--n_classes', default=2, type=int, help='number of classes')
    parser.add_option('--resolution', default=10, type=int, help='resolution of training data')

    (options, args) = parser.parse_args()
    return options


def log_codes():
    with open(os.path.basename(__file__)) as f:
        codes = f.readlines()

    print('\n\n' + '=' * 20 + os.path.basename(__file__) + '=' * 20)
    for c in codes:
        print(c[:-1])

    with open('utils.py', 'r') as f:
        codes = f.readlines()

    print('\n\n' + '=' * 20 + 'utils.py' + '=' * 20)
    for c in codes:
        print(c[:-1])


def get_data_transforms(mean, std):
    out = {
        'train': transforms.Compose([  # 2 steps of data augmentation for training
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]),

        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    }
    return out


if __name__ == '__main__':
    args = get_args()
    log_codes()

    mean = [0.7238, 0.5716, 0.6779]  # for brca
    std = [0.1120, 0.1459, 0.1089]

    data_transforms = get_data_transforms(mean, std)

    print('================================Start loading data!')
    img_trains, img_vals, _ = load_imgs_files(data_path='data', limit=args.N_limit, resolution=args.resolution)
    print('================================Done loading data, train/val: ', len(img_trains), len(img_vals))

    train_set = data_loader(img_trains, transform=data_transforms['train'], APS=args.APS, isTrain=True)
    train_loader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)

    val_set = data_loader(img_vals, transform=data_transforms['val'], APS=args.APS, isTrain=False)
    val_loader = DataLoader(val_set, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

    net = UNet(n_channels=3, n_classes=args.n_classes, bilinear=False)

    if args.gpu:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0, 1])
        cudnn.benchmark = True  # faster convolutions, but more memory
    try:
        train_net(net=net, train_loader=train_loader, val_loader=val_loader, args=args)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED_res{}.pth'.format(args.resolution))
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
