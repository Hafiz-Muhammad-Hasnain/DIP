import cv2
import numpy as np

class MorphOp:
    @staticmethod
    def erode(img, kernel, iterations=1):
        return cv2.erode(img, kernel, iterations=iterations)

    @staticmethod
    def dilate(img, kernel, iterations=1):
        return cv2.dilate(img, kernel, iterations=iterations)

    @staticmethod
    def opening(img, kernel):
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def closing(img, kernel):
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def hit_or_miss(img, kernel):
        return cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)

    @staticmethod
    def gradient(img, kernel):
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

