import cv2
import numpy as np

class StructuringElement:
    @staticmethod
    def create(shape='rect', size=(3, 3)):
        if shape == 'rect':
            return cv2.getStructuringElement(cv2.MORPH_RECT, size)
        elif shape == 'cross':
            return cv2.getStructuringElement(cv2.MORPH_CROSS, size)
        elif shape == 'ellipse':
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
        else:
            return np.ones(size, np.uint8)

    @staticmethod
    def create_hit_or_miss(pattern_type='corner'):
        if pattern_type == 'corner':
            return np.array([[0, 1, 0], [1, -1, 1], [0, 1, 0]], dtype=np.int8)
        return np.ones((3, 3), np.int8)
