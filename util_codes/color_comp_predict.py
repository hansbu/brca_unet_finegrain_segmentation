#!/usr/bin/python

import cv2, os, glob
import numpy as np

colors = np.array([[161, 166, 168], # Gray
                   [255, 255, 0],   # yellow
                   [232, 144, 37],  # Orange
                   [206, 29, 2],  # Red
                   [255, 255, 255],
                  ])


def color_mask(mask):
    uq = np.unique(mask)
    newmask = np.zeros(shape = mask.shape, dtype = np.uint8)
    mask = mask[:,:,1]
    r = newmask[:,:,0]
    g = newmask[:,:,1]
    b = newmask[:,:,2]
    for u in uq:
       r[mask == u] = colors[u,0]
       g[mask == u] = colors[u,1]
       b[mask == u] = colors[u,2]

    newmask = np.dstack((b,g,r))
    return newmask


def colorize(img_path, mask_path):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    print(img_path, mask_path, np.unique(mask))
    mask = color_mask(mask)
    img = np.add(img*0.8, mask*0.2)
    img = cv2.convertScaleAbs(img)
    return img


def color_comp_main(imgsrc, msksrc, dst):
    imglist = sorted(glob.glob(os.path.join(imgsrc, '*.png')))
    msklist = sorted(glob.glob(os.path.join(msksrc, '*.png')))
    print('imglist: {}'.format(len(imglist)))
    print('msklist: {}'.format(len(msklist)))

    for img, msk in zip(imglist, msklist):
        assert os.path.exists(img)
        assert os.path.exists(msk)
        color = colorize(img, msk)
        img_base = os.path.basename(img)
        dstfile = os.path.join(dst, img_base)
        cv2.imwrite(dstfile, color)

