import cv2
import argparse
import math
import numpy as np
import imutils

PARAMS = {
  'thrs1': 143,
  'thrs2': 50,
  'blur': 3,
  'epsilon': 10
}

def experiment_extraction(images):
  imageIndex = 0
  image = cv2.imread(images[imageIndex])

  def slider_moved(x):
    update_image()

  def update_image(save = False):
    params = {}
    for p in PARAMS.keys():
      params[p] = cv2.getTrackbarPos(p, 'experiment')

    if params['blur'] % 2 == 0:
      # blur size must be odd
      params['blur'] = params['blur'] + 1

    print("Update - Params: {}".format(repr(params)))
    canny = prepare_canny_image(image, params)
    output = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

    contours = find_contours(canny, params)
    for c in contours:
      color = (200, 200, 0)
      if contour_filter(c):
        color = (0, 20, 200)

      cv2.drawContours(output, [c], 0, (200, 100, 100), 1)
      rect = cv2.boundingRect(c)
      cv2.rectangle(output, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), color, 1)

    contour = filter_contours(contours)

    if contour is not None:
      rect = cv2.boundingRect(contour)
      print("Contour: {} Rect: {}".format(contour, rect))

      cv2.rectangle(output, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (255,0,), 1)
      cv2.putText(output, images[imageIndex], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

      # extracted = four_point_transform(image, contours)
      extracted = image[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])]
      #extract_subpart(image, contours)
    else:
      cv2.putText(output, "No contour found", (25, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
      extracted = np.zeros((20,135,3), np.uint8)

    cv2.imshow('experiment', imutils.resize(output, height=800))
    (h, w) = extracted.shape[:2]
    cv2.imshow('extracted', cv2.resize(extracted, (w*4,h*4)))

    if save:
      print("Saving images to current folder")
      cv2.imwrite('canny.png', output)
      cv2.imwrite('extracted.png', extracted)

  cv2.namedWindow('experiment')
  cv2.namedWindow('extracted')

  for key in PARAMS.keys():
      cv2.createTrackbar(key, 'experiment', PARAMS[key], 255, slider_moved)

  while (True):
    update_image()
    key = cv2.waitKey(0)

    if key == 27 or key == ord('q'):
      break
    elif key == ord('s'):
      update_image(True)
    elif key == ord('p') or key == 2:
      imageIndex = (imageIndex - 1) % len(images)
      image = cv2.imread(images[imageIndex])
      update_image()
    elif key == ord('n') or key == 3:
      imageIndex = (imageIndex + 1) % len(images)
      image = cv2.imread(images[imageIndex])
      update_image()
    else:
      print("Key pressed: {}".format(key))


  cv2.destroyAllWindows()
  return


def prepare_canny_image(image, params):
  grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurredImage = cv2.GaussianBlur(grayImage, (params['blur'], params['blur']), 0)
  cannyImage = cv2.Canny(blurredImage, params['thrs1'], params['thrs2'])
  return cannyImage

def find_contours(cannyImage, params):
  # find contours in the edge map, then sort them by their
  # size in descending order
  cnts = cv2.findContours(cannyImage.copy(), cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if imutils.is_cv2() else cnts[1]
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

  approxCnts = []
  for cnt in cnts:
    epsilon = params['epsilon'] / 100 * cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    approxCnts.append(approx)

  return cnts # approxCnts

def contour_filter(c):
  rect = cv2.boundingRect(c)
  (w, h) = (rect[2:])
  if w > h and w/h > 6 and w/h < 7:
    return True
  return False

def filter_contours(cnts):
  for c in cnts:
    if contour_filter(c):
      return c

  return None

def extract_digits(image):
  canny = prepare_canny_image(image, PARAMS)
  contour = filter_contours(find_contours(canny))
  if contour is not None:
    rect = cv2.boundingRect(contour)
    extracted = image[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])]
    return extracted
  return None


# https://medium.com/@gsari/digit-recognition-with-opencv-and-python-cbf962f7e2d0
# https://github.com/kazmiekr/GasPumpOCR/blob/master/train_model.py
# https://hackernoon.com/building-a-gas-pump-scanner-with-opencv-python-ios-116fe6c9ae8b

def main():
  parser = argparse.ArgumentParser()

  subparsers = parser.add_subparsers(dest='subparser_name')

  exfd = subparsers.add_parser('experiment-finding-digits')
  exfd.add_argument('images', nargs='+')

  xd = subparsers.add_parser('extract-digits')
  xd.add_argument('image')
  xd.add_argument('output', default=None, nargs='?')


  args = parser.parse_args()

  if args.subparser_name == 'experiment-finding-digits':
    experiment_extraction(args.images)
  elif args.subparser_name == 'extract-digits':
    digits = extract_digits(cv2.imread(args.image))
    if digits is not None:
      if args.output:
        cv2.imwrite(args.output, digits)
      else:
        cv2.imshow('digits', digits)
        cv2.waitKey(0)
    else:
      print('No digits found in {}}', args.image)
  else:
    print("not ready yet to automatically extract")

if __name__ == '__main__':
  main()

