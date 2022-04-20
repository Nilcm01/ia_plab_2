__authors__ = ['1565175', '1566740', ]
__group__ = 'DM.10'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):

        self.neighbors = None
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        # THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """

        self.train_data = train_data.reshape(train_data.shape[0], 14400)
        self.train_data = self.train_data.astype('float64')

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """

        # reshape of test_data array
        test_data = test_data.reshape(test_data.shape[0], 14400)
        test_data = test_data.astype('float64')

        # distances between test_data and train_data
        distances = cdist(test_data, self.train_data, 'euclidean')

        self.neighbors = np.array
        self.neighbors = np.resize(self.neighbors, (test_data.shape[0], k))

        x = np.array
        x = np.resize(x, (test_data.shape[0], k))
        for i in range(distances.shape[0]):
            for j in range(k):
                x[i][j] = np.where(distances[i] == distances[i].min())[0]
                distances[i][x[i][j]] = np.inf

        for i in range(distances.shape[0]):
            for j in range(k):
                self.neighbors[i][j] = self.labels[x[i][j]]

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        most_voted_values = np.array

        for i in self.neighbors:
            dic = dict()
            for index, v in enumerate(i):
                dic[index] = np.count_nonzero(i == v)
            most_voted = max(dic.values())
            most_voted_index = [k for k, v in dic.items() if v == most_voted]
            if len(most_voted_index) != 1:
                x = []
                for j in most_voted_index:
                    x.append(i[j][0])
                most_voted_values = np.append(most_voted_values, min(x))
            else:
                most_voted_values = np.append(most_voted_values, i[most_voted_index][0])
        most_voted_values = np.delete(most_voted_values, 0)
        return most_voted_values

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the class 2nd the  % of votes it got
        """
        self.get_k_neighbours(test_data, k)
        class_predict = self.get_class()

        return class_predict
