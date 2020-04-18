import os, sys, cv2
import multiprocessing as mp
import numpy as np
import random

random.seed(12345)

class gen_annot_predict_pair:
    def __init__(self, annot, predict):
        self.annot_fol = annot
        self.predicted_fol = predict
        self.dest_fol = annot.rstrip('/') + '_combined'
        if not os.path.exists(self.dest_fol):
            os.mkdir(self.dest_fol)
        self.random_pos = True

    def combine(self, fn):
        if self.random_pos:
            fn, index, img_id = fn
            combined_path = os.path.join(self.dest_fol, str(img_id) + '.png')
        else:
            index = 0
            combined_path = os.path.join(self.dest_fol, fn + '.png')

        annot_path = os.path.join(self.annot_fol, fn)
        predicted_path = os.path.join(self.predicted_fol, fn)

        print("combining...", fn)

        annot = cv2.imread(annot_path)
        predicted = cv2.imread(predicted_path)
        dummy = np.zeros((annot.shape[0], 10, 3))
        if index == 0:  # annot on the right
            img = np.hstack((annot, dummy, predicted))
        else:
            img = np.hstack((predicted, dummy, annot))

        cv2.imwrite(combined_path, img)

    def main(self):
        fns = [f for f in os.listdir(self.annot_fol) if f.endswith('.png')]

        if self.random_pos:
            indices = [random.randint(0, 1) for _ in range(len(fns))]
            img_ids = [i for i in range(len(fns))]
            fns = [(f, i, img_id) for f, i, img_id in zip(fns, indices, img_ids)]
            with open(os.path.join(self.dest_fol, 'annot_indices.txt'), 'w') as fid:
                for fn, i, img_id in fns:
                    fid.writelines('{} {} {}\n'.format(fn, img_id, i))

        pool = mp.Pool(processes=20)
        pool.map(self.combine, fns)
        pool.close()


def main_gen_annot_predict_pair(annot, predict):
    gen_annot_predict = gen_annot_predict_pair(annot, predict)
    gen_annot_predict.main()


if __name__ == '__main__':
    annot_fol = sys.argv[1]
    predicted_fol = sys.argv[2]
    main_gen_annot_predict_pair(annot_fol, predicted_fol)



