import cv2
import argparse
import math
import numpy as np
import imutils
from meterreader import DigitalCounterExtraction, DigitsBreaker, DigitsRecognizer

PARAMS = {
  'extraction': {
    'thrs1': 143,
    'thrs2': 50,
    'blur': 3,
    'epsilon': 10
  },
  'breaker': {
    'numdigits': 8
  },
  'recognizer': {
    'blur': 1,
    'thrs1': 0,
    'thrs2': 255
  }
}

def experiment(images):
  extractor = DigitalCounterExtraction(PARAMS['extraction'], True)
  breaker = DigitsBreaker(PARAMS['breaker'], False)
  reader = DigitsRecognizer(PARAMS['recognizer'], True)

  extractor.outputHandler = lambda image: breaker.__setattr__('input', image)
  breaker.outputHandler = lambda image: reader.__setattr__('input', image)
  reader.outputHandler = lambda digits: print("Reco output: {}".format(repr(digits)))

  imageIndex = 0
  extractor.input = images[imageIndex]
  while (True):
    # counter = cv2.cvtColor(counter, cv2.COLOR_BGR2GRAY)
    # allDigits = breaker.process_image(counter)

    # for i in range(8):
    #   reader.process_image(allDigits[i], "digit[{}]".format(i))


    print("waiting for key")
    key = cv2.waitKey(0)
    print("key={}".format(key))

    if key == 27 or key == ord('q'):
      break
    elif key == ord('p') or key == 2:
      imageIndex = (imageIndex - 1) % len(images)
      extractor.input = images[imageIndex]
    elif key == ord('n') or key == 3:
      imageIndex = (imageIndex + 1) % len(images)
      extractor.input = images[imageIndex]
  else:
      print("Key pressed: {}".format(key))

# https://medium.com/@gsari/digit-recognition-with-opencv-and-python-cbf962f7e2d0
# https://github.com/kazmiekr/GasPumpOCR/blob/master/train_model.py
# https://hackernoon.com/building-a-gas-pump-scanner-with-opencv-python-ios-116fe6c9ae8b

def main():
  parser = argparse.ArgumentParser()

  subparsers = parser.add_subparsers(dest='subparser_name')

  exfd = subparsers.add_parser('experiment')
  exfd.add_argument('images', nargs='+')

  xd = subparsers.add_parser('extract-digits')
  xd.add_argument('image')
  xd.add_argument('output', default=None, nargs='?')


  args = parser.parse_args()

  if args.subparser_name == 'experiment':
    experiment(args.images)
  elif args.subparser_name == 'extract-digits':
    pass
    # digits = extract_digits(cv2.imread(args.image))
    # if digits is not None:
    #   process_digits(digits, True)
    #   cv2.waitKey()
    #   cv2.destroyAllWindows()

    #   if args.output:
    #     cv2.imwrite(args.output, digits)
    #   else:
    #     cv2.imshow('digits', digits)
    #     cv2.waitKey(0)
    # else:
    #   print('No digits found in {}}', args.image)
  else:
    print("not ready yet to automatically extract")

if __name__ == '__main__':
  main()

