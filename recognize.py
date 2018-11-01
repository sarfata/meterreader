import cv2
import argparse
import math
import numpy as np
import imutils

# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
  # initialzie a list of coordinates that will be ordered
  # such that the first entry in the list is the top-left,
  # the second entry is the top-right, the third is the
  # bottom-right, and the fourth is the bottom-left
  rect = np.zeros((4, 2), dtype = "float32")

  # the top-left point will have the smallest sum, whereas
  # the bottom-right point will have the largest sum
  s = pts.sum(axis = 1)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]

  # now, compute the difference between the points, the
  # top-right point will have the smallest difference,
  # whereas the bottom-left will have the largest difference
  diff = np.diff(pts, axis = 1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]

  # return the ordered coordinates
  return rect

def four_point_transform(image, pts):
  # obtain a consistent order of the points and unpack them
  # individually
  rect = order_points(pts)
  (tl, tr, br, bl) = rect

  # compute the width of the new image, which will be the
  # maximum distance between bottom-right and bottom-left
  # x-coordiates or the top-right and top-left x-coordinates
  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))

  # compute the height of the new image, which will be the
  # maximum distance between the top-right and bottom-right
  # y-coordinates or the top-left and bottom-left y-coordinates
  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))

  # now that we have the dimensions of the new image, construct
  # the set of destination points to obtain a "birds eye view",
  # (i.e. top-down view) of the image, again specifying points
  # in the top-left, top-right, bottom-right, and bottom-left
  # order
  dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")

  # compute the perspective transform matrix and then apply it
  M = cv2.getPerspectiveTransform(rect, dst)
  warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

  # return the warped image
  return warped

def analyze(imageFilename):
  image = cv2.imread(imageFilename)
  grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  cannyImage = cv2.Canny(grayImage, 200, 250)

  cv2.imshow('GazMeter', cannyImage)
  cv2.waitKey(0)

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

def experiment_extraction(image):
  def slider_moved(x):
    update_image()

  def update_image(save = False):
    paramList = ["thrs1", "thrs2", "blur"]
    params = {}
    for p in paramList:
      params[p] = cv2.getTrackbarPos(p, 'canny')

    output = extract_canny_image(image, params)
    print("Update - Params: {}".format(repr(params)))

    contours = find_contours(output)

    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    if contours is not None:
      rect = cv2.boundingRect(contours)
      print("Contour: {} Rect: {}".format(contours, rect))

      cv2.rectangle(output, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (255,0,), 2)
      cv2.line(output, tuple(contours[0]), tuple(contours[1]), (100,0,255), 2)
      cv2.line(output, tuple(contours[1]), tuple(contours[2]), (100,0,255), 2)
      cv2.line(output, tuple(contours[2]), tuple(contours[3]), (100,0,255), 2)
      cv2.line(output, tuple(contours[3]), tuple(contours[0]), (100,0,255), 2)

      cv2.imshow('canny', imutils.resize(output, height=800))

      # extracted = four_point_transform(image, contours)
      extracted = image[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])]
      #extract_subpart(image, contours)

      (h, w) = extracted.shape[:2]
      cv2.imshow('extracted', cv2.resize(extracted, (w*4,h*4)))

      if save:
        print("Saving images to current folder")
        cv2.imwrite('canny.png', output)
        cv2.imwrite('extracted.png', extracted)
    else:
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(output,'NO CONTOUR',(5,5), font, 4,(255,255,255),2,cv2.LINE_AA)

  cv2.namedWindow('canny')
  cv2.namedWindow('extracted', cv2.WINDOW_NORMAL)
  cv2.createTrackbar('thrs1', 'canny', 121, 255, slider_moved)
  cv2.createTrackbar('thrs2', 'canny', 27, 255, slider_moved)
  cv2.createTrackbar('blur', 'canny', 9, 255, slider_moved)

  while (True):
    update_image()
    key = cv2.waitKey(0)

    if key == 27 or key == ord('q'):
      break
    elif key == ord('s'):
      update_image(True)


  cv2.destroyAllWindows()
  return


def extract_canny_image(image, params):
  grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurredImage = cv2.GaussianBlur(grayImage, (params['blur'], params['blur']), 0)
  cannyImage = cv2.Canny(blurredImage, params['thrs1'], params['thrs2'])
  return cannyImage

def find_contours(cannyImage):
  # find contours in the edge map, then sort them by their
  # size in descending order
  cnts = cv2.findContours(cannyImage.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if imutils.is_cv2() else cnts[1]
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
  displayCnt = None

  # loop over the contours
  for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if the contour has four vertices, then we have found
    # the thermostat display
    if len(approx) == 4:
      displayCnt = approx
      break
  return displayCnt.reshape(4, 2)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('image')
  args = parser.parse_args()

  experiment_extraction(cv2.imread(args.image))
  #rotated = makeImageHorizontal(cv2.imread(args.image))
  #counter = (rotated)
  # cv2.imshow('GazMeter', counter)
  # cv2.waitKey(0)

if __name__ == '__main__':
  main()

