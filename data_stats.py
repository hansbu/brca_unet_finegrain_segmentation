import cv2
import os
import sys
import numpy as np
import collections

fol = 'val_mask_ext'
fns = [f for f in os.listdir(fol) if len(f) > 3]
# print(fns)
H = collections.defaultdict(int)
for i, fn in enumerate(fns):
	mask = cv2.imread(os.path.join(fol, fn), 0)
	unique, counts = np.unique(mask, return_counts=True)
	
	for ind, id in enumerate(unique):
		H[id] += counts[ind]
	print(i, H)