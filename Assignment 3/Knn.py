import csv

class Dataset(object):
	""" Class responsible for interfacing with our data """

	def __init__(self):
		self.X = []
		self.Y = []

	def read_data(self, path):
		with open(path, 'r') as csv_file:
			reader = csv.DictReader(csv_file)
			for row in reader:
				del row['Id']
				self.Y.append(row.pop('Species', None))
				self.X.append(row)

if __name__ == '__main__':
	train = Dataset()
	train.read_data("data/train.csv")
	print '======= Train =========='
	print train.X, '\n\n', train.Y
	print '========================'

	test = Dataset()
	test.read_data("data/test.csv")
	print '======== Test ========='
	print test.X, '\n\n', test.Y
	print '======================='
