import csv
import logging
import re
import numpy as np
import tqdm
from sklearn import svm
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt


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
            self.Y =  np.array(reader[2].split(' ')).astype(np.float)

    def _get_tumor_samples(self, path):
        with open(path, 'r') as inputFile:
            lines = [l.strip().split('	') for l in inputFile.readlines()]
            data = np.matrix(lines[3:]).T
            self.feature_names = data[1]
            data = data[2:]
            data = np.delete(data, list(range(1, data.shape[1], 2)), axis=0)
        self.X = data.astype(np.float16)

    def _describe(self):
        print "\n------ data description -----"
        print "X len = ", len(self.X)
        print "Y len = ", len(self.Y)
        print "# samples = ", self.number_of_samples
        print "# classes = ", self.number_of_classes
        print "-----------------------------\n"

def plot_coefficients(classifier, feature_names, top_features=20):
     coef = classifier.coef_.ravel()
     top_positive_coefficients = np.argsort(coef)[-top_features:]
     top_negative_coefficients = np.argsort(coef)[:top_features]
     top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

     # create plot
     plt.figure(figsize=(30, 15))
     colors = ['#cccccc' if c < 0 else 'teal' for c in coef[top_coefficients]]
     plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
     feature_names = np.array(feature_names)
     plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names, rotation='vertical', ha='right')
     plt.savefig("plot.png")

if __name__ == '__main__':
    training = FeatureData('data/Training_res.txt', 'data/Training_cls.txt')
    test = FeatureData('data/Test_res.txt', 'data/Test_cls.txt')

    training._describe()
    test._describe()

    model = svm.SVC(kernel='linear')
    model.fit(training.X, training.Y)
    results = model.predict(test.X)

    plot_coefficients(model, training.feature_names.tolist()[0])

    for index, result in enumerate(results):
    	print str(result) + " " + str(test.Y[index])
