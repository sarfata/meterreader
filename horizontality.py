import imutils
import numpy as np
import cv2

'''
Return the angle in radians needed to rotate this image and make it straight.
'''
def makeImageHorizontal(image):
  image = imutils.resize(image, height=1000)
  grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurredImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
  cannyImage = cv2.Canny(blurredImage, 1, 200)
  # cv2.imshow('blurred image', cannyImage)
  # cv2.waitKey(0)
  lines = cv2.HoughLines(cannyImage,1,np.pi/180,100)

  print(lines)
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
    cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)

    thetaDeg = theta / np.pi * 180.
    print("Line x0={} Angle={}".format(rho, thetaDeg))
    if thetaDeg > 60 and thetaDeg < 120:
      averageAngles.append(thetaDeg)

  averageAngle = sum(averageAngles) / len(averageAngles)

  rotated = imutils.rotate(image, averageAngle - 90)

  return rotated
