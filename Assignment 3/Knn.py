import csv
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import euclidean
from collections import Counter
import random



class Dataset(object):
	""" Class responsible for interfacing with our data """

	def __init__(self, datafile, K):
		self.X = []
		self.Y = []
		self.num_of_classes = 0
		self.K = K
		with open(datafile, 'r') as data:
			reader = csv.DictReader(data)
			for row in reader:
				self.X.append(map(float, row.values()[1:-1]))
				self.Y.append(row.pop('Species', None))
			self.num_of_classes = len(set(self.Y))

	def step1(self):
		#Get 15 random "Iris-setosa"
		#Use 10 to train and 5 to test
		filtered = [i for i, x in enumerate(self.Y) if x == "Iris-setosa"]
		random.shuffle(filtered)
		Y =  [self.Y[i] for i in filtered]
		X = [self.X[i] for i in filtered]
		self.trainX = X[:10]
		self.trainY = Y[:10]
		self.testX = X[-5:]
		self.testY = Y[-5:]

	def step2(self):
		#Get 15 random "Iris-setosa" and 15 random "Iris-versicolor"
		#Use 10 of each to train and 5 to test
		self.trainX = []
		self.trainY = []
		self.testX = []
		self.testY = []
		for iris in ['Iris-setosa', 'Iris-versicolor']:
			filtered = [i for i, x in enumerate(self.Y) if x == iris]
			random.shuffle(filtered)
			Y =  [self.Y[i] for i in filtered]
			X = [self.X[i] for i in filtered]
			self.trainX.extend(X[:10])
			self.trainY.extend(Y[:10])
			self.testX.extend(X[-5:])
			self.testY.extend(Y[-5:])

	def neighbours(self, test):
		distance = []
		for i in range(len(self.trainY)):
			distance.append(euclidean(self.trainX[i], test))

		return np.argsort(distance)[:self.K]

	def vote(self, n):
		votes = Counter([self.trainY[i] for i in n])
		return votes.most_common(1)[0][0]


	def predict(self):
		print " Actual Class", "       | ", "Predicted Class\n", "--------------------------------------"
		for i in range(len(self.testY)):
			n = self.neighbours(self.testX[i])
			l = 17 - len(self.testY[i])
			print " ", self.testY[i], l*' ', "|   ", self.vote(n)


if __name__ == '__main__':
	train = Dataset('Iris.csv', 3)

	print "\n\n =================\n", 5*" ", "Step 1\n"," =================\n"
	train.step1()
	train.predict()


	print "\n\n =================\n", 5*" ", "Step 2\n"," =================\n"
	train.step2()
	train.predict()















#
