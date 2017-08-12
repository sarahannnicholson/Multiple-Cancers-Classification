#!/usr/bin/env python
import numpy as np
from tqdm import tqdm
import csv, logging, re
from sklearn.svm import SVC
from collections import Counter
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report

class Data(object):
	"""Class responsible for interfacing with our data, eg) getting the data, stats, etc.."""

	def __init__(self, res_path, cls_path, dataType):
		self.dataType = dataType
		self._get_classes(cls_path)
		self._get_tumor_samples(res_path)
		self._clean()

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

	def _clean(self):
		invalid = np.where(np.isin(self.Y, ['14']))[0]
		print invalid
		self.Y = np.delete(self.Y, invalid, 0)
		self.X = np.delete(self.X, invalid, 0)


def feature_selection(X, y, k_val):
	best_indices = SelectKBest(f_classif, k=k_val).fit(X, y).get_support(indices=True)
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

	normalizer = preprocessing.StandardScaler().fit(train.X)
	train.X = normalizer.transform(train.X)
	test.X = normalizer.transform(test.X)
	# ========================
	#    System parameters
	# ========================
	y_train = train.Y
	y_test  = test.Y	
	X_train = train.X
	X_test = test.X

	accuracy = list()
	for x in range(50):
		best_features = set()
		for cls in train.classes:
			features = feature_selection(train.X, train._get_binary(cls), x+1)
			best_features.update(features)

		best =  list(best_features)
		X_train = train.X[:,best]
		X_test = test.X[:,best]

		model = SVC(kernel="linear", probability=True)
		model.fit(X_train, y_train)
		results = model.predict(X_test)
		res = zip(results, y_test)
		
		a =  accuracy_score(y_test, results)
		accuracy.append(a)
		print classification_report(y_test, results)

	print np.max(accuracy)
	print np.argmax(accuracy)


if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)

	train = Data('data/Training_res.txt', 'data/Training_cls.txt', 'train')
	test = Data('data/Test_res.txt', 'data/Test_cls.txt', 'test')

	run_test(train, test)

