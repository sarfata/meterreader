import imutils
import numpy as np
import cv2
from .stage import VisibleStage


class MakeHorizontal(VisibleStage):
    def _process(self):
        debugImage = None
        if self._showYourWork:
            debugImage = self.input.copy()

        originalImage = self.input
        image = imutils.resize(originalImage, height=1000)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurredImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
        cannyImage = cv2.Canny(blurredImage, 1, 200)
        lines = cv2.HoughLines(cannyImage, 1, np.pi/180, 100)

        averageAngles = []
        for l in lines:
            [rho, theta] = l[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            thetaDeg = theta / np.pi * 180.
            if thetaDeg > 60 and thetaDeg < 120:
                averageAngles.append(thetaDeg)

        averageAngle = sum(averageAngles) / len(averageAngles)
        rotated = imutils.rotate(originalImage, averageAngle - 90)

        if self._showYourWork:
            cv2.line(debugImage, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow('Horizontality', rotated)

        self.outputHandler(rotated)

    def initWindows(self):
        cv2.namedWindow('Horizontality')
