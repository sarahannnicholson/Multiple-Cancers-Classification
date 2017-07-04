import logging
import numpy as np
from featureData import FeatureData

class FeatureGenerator(object):
    """Class responsible for generating each feature used in the X matrix."""

    def __init__(self, fd):
        self.classes = fd.classes

    def _get_feature1(self):
        return []

    def _get_feature2(self):
        return []

    def _get_feature3(self):
        return []

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    fd = FeatureData('data/Training_res.txt')
    feature_generator = FeatureGenerator(fd)
