import cv2
import numpy as np
import os.path
from . import VisibleStage
import imutils
from .utils import select_contour, draw_contours, extract_contour
from .recognition import DigitRecognition

'''
input: a list of image
output: a list of cleaned digit
'''


class DigitsCleaner(VisibleStage):
    debugOutputLines = 5
    recognizer = DigitRecognition()

    def _process(self):
        output = []
        if self._showYourWork:
            (self.debugDigitHeight, self.debugDigitWidth) = (20, 17)
            # 20px + image + 20px margin + image + 20px margin + ...
            self.debugWidth = 20 + len(self.input) * \
                (self.debugDigitWidth + 20)
            # 20px top margin + image + 20px margin  (3 times for three lines in the output)
            self.debugHeight = 20 + self.debugOutputLines * \
                (self.debugDigitHeight + 20)
            self.debugImage = np.zeros(
                (self.debugHeight, self.debugWidth, 3), dtype=np.uint8)

        for index, image in enumerate(self.input):
            output.append(self.process_digit(image, index))

        if self._showYourWork:
            cv2.imshow('DigitsCleaner', cv2.resize(
                self.debugImage, (self.debugWidth*3, self.debugHeight*3)))

        self.outputHandler(output)

    def process_digit(self, image, index):
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.params['blur'] > 0:
            blur = cv2.GaussianBlur(
                grayImage, (self.params['blur'], self.params['blur']), 0)
        else:
            blur = grayImage
        # +cv2.THRESH_OTSU - overrides default
        _, threshold = cv2.threshold(
            blur, self.params['thrs1'], self.params['thrs2'], cv2.THRESH_BINARY)

        (imageHeight, imageWidth) = image.shape[:2]

        def filter_contour(cnt):
            rect = cv2.boundingRect(cnt)
            (w, h) = (rect[2:])
            # reject contours that take the entire width or height
            if h > 0.9 * imageHeight or w > 0.9 * imageWidth:
                return False
            # reject contours where height is not at least twice the width
            if h < w:
                return False
            # reject images that are less than 1/2 the height
            if h < 0.3 * imageHeight:
                return False
            return True

        # Make sure the image is big enough ...
        if imageHeight < 30 or imageWidth < 20:
            if self._showYourWork:
                self.draw_debug_image(index, 0, image)
                self.draw_debug_image(index, 1, blur)
                self.draw_debug_image(index, 2, threshold)

                self.draw_result(index, 'X')

            return None

        # Find contours in the image.
        cnts, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # Select one of the contours
        best_contour = select_contour(cnts, filter_contour)

        # Extract the content of this contour
        digit = extract_contour(threshold, best_contour)

        # Make sure we always have the same output size
        digit = cv2.resize(digit, (40, 36))

        if self._showYourWork:
            self.draw_debug_image(index, 0, image)
            self.draw_debug_image(index, 1, blur)
            self.draw_debug_image(index, 2, threshold)
            self.draw_debug_image(index, 3, draw_contours(
                threshold, cnts, filter_contour))
            #self.draw_debug_image(index, 4, digit)

            self.draw_result(index, self.recognizer.recognize(digit))

        return digit

    def _initWindows(self):
        cv2.namedWindow('DigitsCleaner')
        cv2.createTrackbar('blur', 'DigitsCleaner',
                           self._params['blur'], 5, self.guicb)
        cv2.createTrackbar('thrs1', 'DigitsCleaner',
                           self._params['thrs1'], 255, self.guicb)
        cv2.createTrackbar('thrs2', 'DigitsCleaner',
                           self._params['thrs2'], 255, self.guicb)

    def guicb(self, value):
        print(repr(value))
        params = self._params
        params['blur'] = cv2.getTrackbarPos('blur', 'DigitsCleaner')
        if params['blur'] != 0 and params['blur'] % 2 != 1:
            params['blur'] = params['blur'] + 1
            cv2.setTrackbarPos('blur', 'DigitsCleaner', params['blur'])

        params['thrs1'] = cv2.getTrackbarPos('thrs1', 'DigitsCleaner')
        params['thrs2'] = cv2.getTrackbarPos('thrs2', 'DigitsCleaner')
        self.params = params

    def draw_debug_image(self, indexX, indexY, image):
        if len(image.shape) != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        x = 20+indexX*(self.debugDigitWidth+20)
        y = 20+indexY*(self.debugDigitHeight+20)

        self.debugImage[y:y+image.shape[0], x:x+image.shape[1]] = image

    def draw_result(self, indexX, result):
        x = 20+indexX*(self.debugDigitWidth+20)
        y = self.debugOutputLines * (self.debugDigitHeight+20) + 10
        cv2.putText(self.debugImage, str(result), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
