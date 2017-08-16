#!/usr/bin/env python
import numpy as np
import itertools
from tqdm import tqdm
import csv, logging, re
from sklearn.svm import SVC
from collections import Counter
from sklearn import preprocessing
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

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

def plot_coefficients(classifier, feature_names, class_names, top_features=10):
	for x in range(len(class_names[1:])):
		coef = classifier.coef_[x]

		top_positive_coefficients = np.argsort(coef)[-top_features:]
		top_negative_coefficients = np.argsort(coef)[:top_features]
		top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

		# create plot
		plt.figure(figsize=(30, 15))
		colors = ['#cccccc' if c < 0 else 'teal' for c in coef[top_coefficients]]
		plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
		names = np.array(feature_names)[top_coefficients]
		plt.xticks(np.arange(1, 1 + 2 * top_features), names, rotation='vertical', ha='right')
		plt.savefig("graphs/plot" + class_names[x] + ".png")

def print_accuracy(accuracy, num_samples, val_accuracy):

	val_mean = np.mean(val_accuracy, axis=1)
	val_std = np.std(val_accuracy, axis=1)


	plt.figure()
	plt.plot(num_samples, accuracy, color="b")
	plt.plot(num_samples, val_mean, color="g")
	plt.fill_between(num_samples,
		val_mean - val_std,
		val_mean + val_std,
		alpha=0.3,
		color="g")

	plt.title("Learning Graph")
	plt.ylabel('Accuracy')
	plt.xlabel('# Of Samples')
	plt.savefig("graphs/learning.png")

def print_confusion(y_test, y_pred, classes):
	plt.figure()
	cmap=plt.cm.Blues
	cm = confusion_matrix(y_test, y_pred)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title("Confusion Matrix")
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=90)
	plt.yticks(tick_marks, classes)

	fmt = '.1f'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig("graphs/confusion.png")

def train_test(train_X, train_y, test_X, test_y, classes):
	best_features = set()
	for cls in classes:
		index = classes.index(cls) - 1
		y_bin =  [c == str(index) for c in train_y]
		features = feature_selection(train_X, y_bin, 43)
		best_features.update(features)

	best =  list(best_features)
	X_train = train_X[:,best]
	X_test = test_X[:,best]

	model = SVC(kernel="linear")
	model.fit(X_train, train_y)
	results = model.predict(X_test)

	return  accuracy_score(test_y, results)

def run_test(train, test):
	train._describe()
	test._describe()

	normalizer = preprocessing.StandardScaler().fit(train.X)
	train.X = normalizer.transform(train.X)
	test.X = normalizer.transform(test.X)
	# ========================
	#	System parameters
	# ========================
	y_train = train.Y
	y_test  = test.Y	
	X_train = train.X
	X_test = test.X

	X_train_s, y_train_s = shuffle(train.X, train.Y, random_state=4)

	accuracy = list()
	num_samples = list()


	for i in range(10, len(y_train_s), 5):
		train_X = X_train_s[:i]
		train_y = y_train_s[:i]

		a = train_test(train_X, train_y, X_test, y_test, train.classes)

		accuracy.append(a)
		num_samples.append(i)
	print accuracy

	val_accuracy = list()
	val_num_samples = list()

	for i in range(10, len(y_train), 5):
		tmp_list = list()
		for z in range(10):
			X_train, X_test, y_train, y_test = train_test_split(
			train.X, train.Y, train_size=i)
			a = train_test(X_train, y_train, X_test, y_test, train.classes)

			tmp_list.append(a)
		val_accuracy.append(tmp_list)
		val_num_samples.append(i)
	print accuracy
	print_accuracy(accuracy, num_samples, val_accuracy)

	#plot_coefficients(model, [train.feature_names[i] for i in best_features], train.classes)

	#print_confusion(y_test, results, train.classes[1:])


if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)

	train = Data('data/Training_res.txt', 'data/Training_cls.txt', 'train')
	test = Data('data/Test_res.txt', 'data/Test_cls.txt', 'test')

	run_test(train, test)

