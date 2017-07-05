import csv
import logging
import re
import numpy as np
import tqdm
from sklearn import svm


class FeatureData(object):
    """Class responsible for interfacing with our data, eg) getting the data, stats, etc.."""

    def __init__(self, res_path, cls_path):
        self._get_classes(cls_path)
       	self._get_tumor_samples(res_path)

    def _get_classes(self, path):
    	with open(path, 'r') as f:
            reader = [l.strip() for l in f.readlines()]
            self.number_of_samples = reader[0].split(' ')[0]
            self.number_of_classes = reader[0].split(' ')[1]
            self.classes = reader[1].split(' ')
            self.Y = reader[2].split(' ')

    def _get_tumor_samples(self, path):
        with open(path, 'r') as inputFile:
            lines = [l.strip().split('	') for l in inputFile.readlines()]           
            data = np.matrix(lines[3:]).T[2:]
            data = np.delete(data, list(range(1, data.shape[1], 2)), axis=0)

        self.X = data.astype(float)

    def _describe(self):
        print len(self.X)
        print len(self.Y)
        print self.number_of_samples
    	print self.number_of_classes


if __name__ == '__main__':
    training = FeatureData('data/Training_res.txt', 'data/Training_cls.txt')
    test = FeatureData('data/Test_res.txt', 'data/Test_cls.txt')

    training._describe()
    test._describe()

    model = svm.SVC()
    model.fit(training.X, training.Y)

    results = model.predict(test.X)

    for index, result in enumerate(results):
    	print str(result) + " " + str(test.Y[index])

