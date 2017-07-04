import csv
import logging
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import feature_extraction
import tqdm


class FeatureData(object):
    """Class responsible for interfacing with our data, eg) getting the data, stats, etc.."""

    def __init__(self, file_path):
        self.number_of_classes = 14
        self.classes = ["Breast", "Prostate", "Lung", "Colorectal", "Lymphoma", "Bladder", "Melanoma",
                        "Uterus__Adeno", "Leukemia", "Renal", "Pancreas", "Ovary", "Mesothelioma", "CNS"]
        self.tumor_samples = self._get_tumor_samples(file_path)
        self.number_of_samples = len(self.tumor_samples)

    def _get_tumor_samples(self, path):
        # Body ID, articleBody
        samples = []
        with open(path, 'r') as inputFile:
            reader = csv.reader(inputFile, 'excel-tab')
            cnt =0
            for row in reader:
                if cnt < 5:
                    row_stripped = filter(lambda name: name.strip(), row)
                    print row_stripped, "\n\n"
                    samples.append(row_stripped)
                cnt +=1
        return samples



if __name__ == '__main__':
    fd = FeatureData('data/Training_res.txt')

    cnt = 1
    for x in fd.tumor_samples:
        print "line ", cnt, " = ", len(x), "\n\n"
        cnt +=1
