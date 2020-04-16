import os, sys, cv2
import multiprocessing as mp
import numpy as np


class gen_annot_predict_pair:
    def __init__(self, annot, predict):
        self.annot_fol = annot
        self.predicted_fol = predict
        self.dest_fol = annot.rstrip('/') + '_combined'

    def combine(self, fn):
        annot_path = os.path.join(self.annot_fol, fn)
        predicted_path = os.path.join(self.predicted_fol, fn)
        combined_path = os.path.join(self.dest_fol, fn)

        print(fn)

        annot = cv2.imread(annot_path)
        predicted = cv2.imread(predicted_path)
        dummy = np.zeros((annot.shape[0], 10, 3))

        img = np.hstack((annot, dummy, predicted))
        cv2.imwrite(combined_path, img)

    def main(self):
        fns = [f for f in os.listdir(self.annot_fol) if f.endswith('.png')]
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



