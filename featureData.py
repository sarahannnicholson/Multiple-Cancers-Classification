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
			self.classes = reader[1].split(' ')[0:]
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

	def _get_binary(self, name):
		try:
			index = self.classes.index(name) - 1
			return  [c == str(index) for c in self.Y]
		except ValueError:
			return False

def run_test(train, test):
	train._describe()
	test._describe()

	for c in test.classes[1:]:
		print c
		trainY = train._get_binary(c)
		testY = test._get_binary(c)

		if not trainY or not testY:
			print "Not enough data"
			continue

		model = svm.SVC(kernel='linear')
		model.fit(train.X, trainY)
		results = model.predict(test.X)
		res = zip(results, testY)
		truePos = np.count_nonzero([y[0] for y in res if y[1]])
		falsePos = np.count_nonzero([y[0] for y in res if not y[1]])
		falseNeg = np.count_nonzero([not y[0] for y in res if y[1]])
		print "T+" + str(truePos)
		print "F+" + str(falsePos)
		print "F-" + str(falseNeg)


if __name__ == '__main__':
	train = FeatureData('data/Training_res.txt', 'data/Training_cls.txt')
	test = FeatureData('data/Test_res.txt', 'data/Test_cls.txt')

	run_test(train, test)
	