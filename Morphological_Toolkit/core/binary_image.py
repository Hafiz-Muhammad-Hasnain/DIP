import cv2
import numpy as np

class BinaryImage:
    def __init__(self, path=None):
        self.image = None
        self.binary = None
        if path:
            self.load(path)

    def load(self, path):
        self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return self.image is not None

    def binarize(self, method='otsu', threshold=127):
        if self.image is None:
            return None
            
        if method == 'otsu':
            _, self.binary = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            _, self.binary = cv2.threshold(self.image, threshold, 255, cv2.THRESH_BINARY)
            
        return self.binary
