import csv
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class Dataset(object):
	""" Class responsible for interfacing with our data """

	def __init__(self):
		self.X = []
		self.Y = []
		self.num_of_classes = 0

	def read_data(self, path):
		with open(path, 'r') as csv_file:
			reader = csv.DictReader(csv_file)
			for row in reader:
				del row['Id']
				self.Y.append(row.pop('Species', None))
				self.X.append(map(float, row.values()))
		self.num_of_classes = len(set(self.Y))


def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
	# train on the input data
	train(X_train, y_train)

	# loop over all observations
	for i in range(len(X_test)):
		predictions.append(predict(X_train, y_train, X_test[i, :], k))

if __name__ == '__main__':
	train = Dataset()
	train.read_data("data/train.csv")
	print '======= Train =========='
	print train.X, '\n'
	print train.Y, '\n'
	print 'classes =', train.num_of_classes
	print '========================'

	test = Dataset()
	test.read_data("data/test.csv")
	print '======== Test ========='
	print test.X, '\n'
	print test.Y, '\n'
	print 'classes =' , test.num_of_classes
	print '======================= \n'

	# Train model
	clf = KNeighborsClassifier(len(train.X))
	clf.fit(train.X, train.Y)

	# Test model
	pred_Y = clf.predict(test.X)

	# evaluate accuracy
	print "accuracy = ", accuracy_score(test.Y, pred_Y)














#
