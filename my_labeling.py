__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import numpy as np
import Kmeans
import KNN
from KNN import *
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
#import cv2

if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    # List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))


# You can start coding your functions here
def retrieval_by_shape(imgs, labels, pregunta):
    index = np.where(labels == pregunta)
    return imgs[index[0]]


def get_shape_accuracy(labels, ground_truth):
    correctes = (labels == ground_truth)
    return 100*sum(correctes)/len(correctes)


knn = KNN(train_imgs, train_class_labels)
preds = knn.predict(test_imgs, 2)

imgs = retrieval_by_shape(test_imgs, preds, 'Flip Flops')
visualize_retrieval(imgs, len(imgs))

percent_correctes = get_shape_accuracy(preds, test_class_labels)
print(percent_correctes)