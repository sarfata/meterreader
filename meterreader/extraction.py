import cv2
import argparse
import math
import numpy as np
import imutils
from . import VisibleStage

'''
A 'Stage' transformation:
 - input: an image name to load from disk
 - output: an image of the gazmeter counter
'''
class DigitalCounterExtraction(VisibleStage):
  def _initWindows(self):
    cv2.namedWindow('Extraction')
    cv2.namedWindow('extracted')
    cv2.createTrackbar('blur', 'Extraction', self._params['blur'], 15, self.guicb)
    cv2.createTrackbar('thrs1', 'Extraction', self._params['thrs1'], 255, self.guicb)
    cv2.createTrackbar('thrs2', 'Extraction', self._params['thrs2'], 255, self.guicb)
    cv2.createTrackbar('epsilon', 'Extraction', self._params['epsilon'], 255, self.guicb)

  def guicb(self, value):
    print(repr(value))
    params = self._params
    params['blur'] = cv2.getTrackbarPos('blur', 'Extraction')
    if params['blur'] != 0 and params['blur'] % 2 != 1:
      params['blur'] = params['blur'] + 1
      cv2.setTrackbarPos('blur', 'Extraction', params['blur'])

    params['thrs1'] = cv2.getTrackbarPos('thrs1', 'Extraction')
    params['thrs2'] = cv2.getTrackbarPos('thrs2', 'Extraction')
    params['epsilon'] = cv2.getTrackbarPos('epsilon', 'Extraction')
    self.params = params

  def _process(self):
    image = cv2.imread(self.input)

    canny = self.prepare_canny_image(image)
    contours = self.find_contours(canny)
    contour = self.select_best_contour(contours)

    extracted = None
    if contour is not None:
      rect = cv2.boundingRect(contour)
      extracted = image[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])]
    else:
      print("Error: {} Unable to find contour of digits in image".format(self.input))

    # We are done - Do some presentation of the output
    if self._showYourWork:
      # Make output image a copy of the canny image with colors and then draw
      # contours in it.
      output = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
      cv2.putText(output, self.input, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
      self.show_contours(contours, output)

      # Draw contour in the output
      if contour is not None:
        rect = cv2.boundingRect(contour)
        cv2.rectangle(output, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (10,200,0), 2)
      else:
        cv2.putText(output, "No contour found", (25, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)

      # Draw the output image and the extracted counter in a separate window.
      cv2.imshow('Extraction', imutils.resize(output, height=800))
      if extracted is not None:
        (h, w) = extracted.shape[:2]
        cv2.imshow('extracted', cv2.resize(extracted, (w*4,h*4)))
      else:
        cv2.imshow('extracted', np.zeros((20*4, 135*4, 3)))

    self.outputHandler(extracted)

  # Step1: Transform image to B&W and extract contrast lines
  def prepare_canny_image(self, image):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurredImage = cv2.GaussianBlur(grayImage, (self.params['blur'], self.params['blur']), 0)
    cannyImage = cv2.Canny(blurredImage, self.params['thrs1'], self.params['thrs2'])
    return cannyImage

  # Step2: Identify contours in image to isolate objects
  def find_contours(self, cannyImage):
    # find contours in the edge map
    cnts = cv2.findContours(cannyImage.copy(), cv2.RETR_TREE,
      cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    return cnts

  # Step3: Select the best contour in the image
  def select_best_contour(self, cnts):
    # sort contours
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # keep the first one that matches our criteria
    for c in cnts:
      if self.contour_filter(c):
        return c

  # Utility function used to decide if a contour is a candidate or not.
  # we use some basic heuristic here to eliminate all unlikely contours.
  def contour_filter(self, c):
    rect = cv2.boundingRect(c)
    (w, h) = (rect[2:])
    # be even more specific to reject wrong contours...
    if h < 20 or h > 25 or w < 130 or w > 140:
      return False
    if w > h and w/h > 6 and w/h < 7:
      return True
    return False

  # A function to draw all the contours on an output image.
  def show_contours(self, contours, output):
    for c in contours:
      color = (200, 200, 0)
      if self.contour_filter(c):
        color = (0, 20, 200)
      cv2.drawContours(output, [c], 0, (200, 100, 100), 1)
      rect = cv2.boundingRect(c)
      cv2.rectangle(output, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), color, 1)



# def experiment_extraction(images):
#   imageIndex = 0
#   image = cv2.imread(images[imageIndex])

#   def slider_moved(x):
#     update_image()

#   for key in PARAMS.keys():
#       cv2.createTrackbar(key, 'experiment', PARAMS[key], 255, slider_moved)


