import cv2
import numpy as np

'''
Select first contour in a list that passes a filter function.
'''
def select_contour(cnts, cntFilter):
  for c in cnts:
    if cntFilter(c):
      return c
  return None

'''
Takes a B&W image, turns it into color and draws a list of contours on it.

Contours will be blue if they pass a filter, red otherwise.
'''
def draw_contours(original, cnts, cntFilter):
  newImage = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
  for cnt in cnts:
    rect = cv2.boundingRect(cnt)
    color = (0, 20, 200)
    if cntFilter(cnt):
      color = (200, 200, 0)
    cv2.rectangle(newImage, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), color, 1)
  return newImage

'''
Take an image, extract the part defined by the rect(contour), move it to the top left
corner and return a new image.
'''
def extract_contour(image, contour):
  rect = cv2.boundingRect(contour)
  extracted = image[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])]

  newImage = image.copy()
  newImage[:,:] = (0)

  if rect[2] > 0 and rect[3] > 0:
    newImage[0:rect[3], 0:rect[2]] = extracted

  return newImage