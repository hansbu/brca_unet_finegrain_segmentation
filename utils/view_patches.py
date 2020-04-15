import numpy as np
import cv2
import os
import sys


fol = 'train'
# fol = sys.argv[1]

out = 'out_' + fol
if not os.path.isdir(out):
	os.mkdir(out)


fns = [f for f in os.listdir(fol + '_jpg_ext') if len(f) > 3]
# print(fns)
for i, fn in enumerate(fns):
	print(i, fn)
	img_path = os.path.join(fol + '_jpg_ext', fn)
	img_mask = os.path.join(fol + '_mask_ext', fn.split('.')[0] + '.png')
	img  = cv2.imread(img_path)
	mask = cv2.imread(img_mask, 0)

	# Color - Index - Label
	# Yellow - 0 - Gleason 3: BRG: 
	# Orange - 1 - Gleason 4
	# Red    - 2 - Gleason 5
	# Gray   - 3 - Benign
	# White  - 4 - Stroma
	x = img*0 + 255
	for row in range(mask.shape[1]):
		for col in range(mask.shape[0]):
			px = mask[row, col]
			if px == 0:  # yellow
				x[row, col] = np.array([0, 255, 255])
			elif px == 1:	# orange
				x[row, col] = np.array([0, 165, 255])
			elif px == 2:	# red
				x[row, col] = np.array([0, 0, 255])
			elif px == 3: 	# gray
				x[row, col] = np.array([128, 128, 128])

	x = x.astype(np.uint8)

	img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
	x = cv2.resize(x, (0, 0), fx = 0.5, fy = 0.5)
	# mask = mask*55
	# mask = mask.astype(np.uint8)

	img = np.hstack((img, x))
	cv2.imwrite(os.path.join(out, fn.split('.')[0] + '.png'), img)

	# cv2.imshow('img', img)
	# cv2.waitKey(0)
