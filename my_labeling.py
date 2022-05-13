import unittest
import pickle
import numpy as np
import KNN as k
from KNN import *
from utils import *
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
# import cv2
from skimage import io


def retrieval_by_shape(imgs, labels, pregunta):
    index = np.where(labels == pregunta)
    return imgs[index[0]]


class TestCases(unittest.TestCase):

    def setUp(self):
        # Load all the images and GT
        self.train_imgs, self.train_class_labels, self.train_color_labels, \
        self.test_imgs, self.test_class_labels, self.test_color_labels = read_dataset(ROOT_FOLDER='./images/',
                                                                       gt_json='./images/gt.json')
        # List with all the existant classes
        self.classes = list(set(list(self.train_class_labels) + list(self.test_class_labels)))

        np.random.seed(123)
        with open('./test/test_cases_knn.pkl', 'rb') as f:
            self.test_cases = pickle.load(f)

    def test_retrieval_by_shape(self):
        imgs = retrieval_by_shape(self.test_imgs, self.test_class_labels, 'Flip Flops')
        visualize_retrieval(imgs, len(imgs))

if __name__ == "__main__":
    unittest.main()
