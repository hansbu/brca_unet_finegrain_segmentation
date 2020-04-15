import os
import sys
import numpy as np
import cv2
from color_comp_func import *

train_imgs = []

# in_imgs_fol = 'val_jpg_ext'
# in_mask_fol = 'val_mask_ext'
# out_imgs_fol = 'val_jpg_10x'
# out_mask_fol = 'val_mask_10x'

in_imgs_fol = 'train_jpg_ext'
in_mask_fol = 'train_mask_ext'
out_imgs_fol = 'train_jpg_10x'
out_mask_fol = 'train_mask_10x'

for fol in [out_imgs_fol, out_mask_fol]:
    if not os.path.exists(fol): os.mkdir(fol)

# out_colored_fol = 'colored_val_10x'

img_fns = [os.path.join(in_imgs_fol, f) for f in os.listdir(in_imgs_fol) if len(f) > 3]

c = 0
for fn in img_fns:
    mask_fn = os.path.join(in_mask_fol, fn.split('/')[-1].split('.jpg')[0] + '.png')
    img = cv2.imread(fn)
    mask = cv2.imread(mask_fn, 0)
    img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
    mask = cv2.resize(mask, (0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_NEAREST)

    # print(mask.shape)
    # print(mask.min(), mask.max())
    mask[mask == 2] = 1

    cv2.imwrite(os.path.join(out_imgs_fol, fn.split('/')[-1].split('.jpg')[0]) + '.png', img)
    cv2.imwrite(os.path.join(out_mask_fol, mask_fn.split('/')[-1]), mask)

    c += 1
    if c % 10 == 0: print('Processing: {}/{}'.format(c, len(img_fns)))

# color_comp_main(out_imgs_fol, out_mask_fol, out_colored_fol)
