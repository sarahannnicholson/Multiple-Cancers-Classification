#!/usr/bin/env python
import numpy as np
from tqdm import tqdm
import csv, logging, re
from sklearn.svm import SVC
from collections import Counter
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

class Data(object):
	"""Class responsible for interfacing with our data, eg) getting the data, stats, etc.."""

	def __init__(self, res_path, cls_path, dataType):
		self.dataType = dataType
		self._get_classes(cls_path)
		self._get_tumor_samples(res_path)

	def _get_classes(self, path):
		print "Getting " + self.dataType + " classes"
		with open(path, 'r') as f:
			reader = [l.strip() for l in tqdm(f.readlines())]
			self.number_of_samples = reader[0].split(' ')[0]
			self.number_of_classes = reader[0].split(' ')[1]
			self.classes = reader[1].split(' ')[0:]
			self.Y = reader[2].split(' ')

	def _get_tumor_samples(self, path):
		print "Getting " + self.dataType + " samples"
		with open(path, 'r') as inputFile:
			lines = [l.strip().split('	') for l in tqdm(inputFile.readlines())]
			data = np.matrix(lines[3:])
			self.feature_names = data[:,1]
			data = data[:,2:]
			data = np.delete(data, list(range(1, data.shape[1], 2)), axis=1)
		self.X = data.astype(float).T

	def _get_binary(self, name):
		try:
			index = self.classes.index(name) - 1
			return  [c == str(index) for c in self.Y]
		except ValueError:
			return False

	def _describe(self):
		print "\n------ data " + self.dataType + " description -----"
		print "X len = ", len(self.X)
		print "Y len = ", len(self.Y)
		print "# samples = ", self.number_of_samples
		print "# classes = ", self.number_of_classes
		print "-----------------------------\n"

# Failed attempt to do feature selection
def feature_selection_F(data):
	new_X = data[0] # hackzz
	with open('bestFeatures.txt', 'r') as f:
		lines = [l.strip().split(',') for l in f.readlines()]
		gene_list = [gene for line in lines for gene in line]
	for gene in gene_list:
		rows = np.where(data == gene)
		new_X = np.append(new_X, data[rows[0]], axis=0)
	print new_X
	return new_X[1:] # Eww

def feature_selection(X, y):
	best_indices = SelectKBest(mutual_info_classif, k=2).fit(X, y).get_support(indices=True)
	return best_indices

def plot_coefficients(classifier, feature_names, class_name, top_features=20):
	 coef = classifier.coef_[0]

	 top_positive_coefficients = np.argsort(coef)[-top_features:]
	 top_negative_coefficients = np.argsort(coef)[:top_features]
	 top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

	 # create plot
	 plt.figure(figsize=(30, 15))
	 colors = ['#cccccc' if c < 0 else 'teal' for c in coef[top_coefficients]]
	 plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
	 feature_names = np.array(feature_names)[top_coefficients]
	 plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names, rotation='vertical', ha='right')
	 plt.savefig("graphs/plot" + class_name + ".png")


def run_test(train, test):
	train._describe()
	test._describe()
	doFeatureSelection = False

	print "Train and test model"
	for c in tqdm(test.classes[1:]):
		y_train = train._get_binary(c)
		y_test  = test._get_binary(c)

		if not y_train or not y_test:
			print "Not enough data"
			continue
		
		if doFeatureSelection: 	# Best features
			best_indices = feature_selection(train.X, y_train)
			X_train = train.X[:,best_indices]
			X_test = test.X[:,best_indices]
		else: #without best features
			X_train = train.X
			X_test = test.X

		model = SVC(kernel="linear")
		model.fit(X_train, y_train)
		#plot_coefficients(model, train.feature_names.tolist()[0], c)
		results = model.predict(X_test)
		res = zip(results, y_test)
		truePos = np.count_nonzero([y[0] for y in res if y[1]])
		falsePos = np.count_nonzero([y[0] for y in res if not y[1]])
		falseNeg = np.count_nonzero([not y[0] for y in res if y[1]])
		print c
		#print float(truePos) / (truePos + falseNeg)
		print truePos
		print "T+" + str(truePos)
		print "F+" + str(falsePos)
		print "F-" + str(falseNeg)


if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)

	train = Data('data/Training_res.txt', 'data/Training_cls.txt', 'train')
	test = Data('data/Test_res.txt', 'data/Test_cls.txt', 'test')

	run_test(train, test)
