#!/usr/bin/env python3

import cv2
import argparse
import math
import numpy as np
import imutils
import sys
import os.path
import glob
import random
from meterreader import DigitalCounterExtraction, DigitsBreaker, DigitsCleaner, DigitsRecognizer, DigitRecognition

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
  'cleaner': {
    'blur': 1,
    'thrs1': 0,
    'thrs2': 255
  },
  'recognizer': {

  }
}

def experiment(images):
  extractor = DigitalCounterExtraction(PARAMS['extraction'], True)
  breaker = DigitsBreaker(PARAMS['breaker'], False)
  cleaner = DigitsCleaner(PARAMS['cleaner'], True)
  recognizer = DigitsRecognizer(PARAMS['recognizer'])

  extractor.outputHandler = lambda image: breaker.__setattr__('input', image)
  breaker.outputHandler = lambda image: cleaner.__setattr__('input', image)
  cleaner.outputHandler = lambda digits: recognizer.__setattr__('input', digits)
  recognizer.outputHandler = lambda digits: print("Reco output: {}".format(repr(digits)))

  imageIndex = 0
  extractor.input = images[imageIndex]
  while (True):
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

def extract(image):
  extractor = DigitalCounterExtraction(PARAMS['extraction'], True)
  breaker = DigitsBreaker(PARAMS['breaker'], False)
  cleaner = DigitsCleaner(PARAMS['cleaner'], True)
  recognizer = DigitsRecognizer(PARAMS['recognizer'])

  output = { 'value': None }
  def outputHandler(v):
    output['value'] = "".join(map(str, v))

  extractor.outputHandler = lambda image: breaker.__setattr__('input', image)
  breaker.outputHandler = lambda image: cleaner.__setattr__('input', image)
  cleaner.outputHandler = lambda image: recognizer.__setattr__('input', image)
  recognizer.outputHandler = outputHandler

  extractor.input = image
  return output['value']

def labelSamples(folder):
  fileList = glob.glob('{}/*jpg'.format(folder))
  random.shuffle(fileList)
  for filename in fileList:
    print(filename)
    if os.path.isfile(filename + ".txt"):
      continue

    cv2.imshow('image', cv2.imread(filename))

    reading = []
    key = cv2.waitKey(0)
    while key != 13:
      if key >= ord('0') and key <= ord('9'):
        reading.append(int(chr(key)))
        print(reading)
      if key == ord('q'):
        sys.exit(0)
      key = cv2.waitKey(0)

    with open(filename + ".txt", 'w') as f:
      f.write("".join(map(str, reading)))

def trainWithSamples(folder):
  fileList = glob.glob('{}/*jpg'.format(folder))
  extractor = DigitalCounterExtraction(PARAMS['extraction'], False)
  breaker = DigitsBreaker(PARAMS['breaker'], False)
  cleaner = DigitsCleaner(PARAMS['cleaner'], False)
  recognition = DigitRecognition()

  extractor.outputHandler = lambda image: breaker.__setattr__('input', image)
  breaker.outputHandler = lambda image: cleaner.__setattr__('input', image)

  counter = {}
  hogdata = []
  results = []

  for filename in fileList:
    # skip files for which we have no label
    if not os.path.isfile(filename + ".txt"):
      continue
    result = list(open(filename + ".txt").read())
    if len(result) != 8:
      print("{}: Skipping because of invalid expected results: {}".format(filename, result))
      continue

    def outputHandler(digitsImages):
      for image in digitsImages:
        hog =recognition.hog_of_digit(image)
        expected = result.pop(0)

        if len(hog) == 81:
          hogdata.append(hog)
          results.append(expected)
          if not expected in counter:
            counter[expected] = 0
          counter[expected] = counter[expected] + 1
          print("{}: added {} samples".format(filename, len(digitsImages)))
        else:
          print("{}: error - invalid HOG - len={}".format(filename, len(hog)))

    cleaner.outputHandler = outputHandler
    try:
      extractor.input = filename
    except Exception as e:
      print("{}: Error {}".format(filename, e))

  total = sum(counter.values())
  for k in sorted(counter.keys()):
    print("{}: {} - {:.0%}".format(k, counter[k], counter[k]/total))

  svm = cv2.ml.SVM_create()
  svm.setKernel(cv2.ml.SVM_RBF)
  trainingData = np.float32(hogdata).reshape(-1,81)
  trainingResult = np.asarray(results, dtype=int)[:,np.newaxis]
  svm.trainAuto(trainingData, cv2.ml.ROW_SAMPLE, trainingResult)
  svm.save('svm_data.dat')
  print("SVM Model saved!")

def testSamples(folder):
  extractor = DigitalCounterExtraction(PARAMS['extraction'], False)
  breaker = DigitsBreaker(PARAMS['breaker'], False)
  cleaner = DigitsCleaner(PARAMS['cleaner'], False)
  recognizer = DigitsRecognizer(PARAMS['recognizer'])

  extractor.outputHandler = lambda image: breaker.__setattr__('input', image)
  breaker.outputHandler = lambda image: cleaner.__setattr__('input', image)
  cleaner.outputHandler = lambda image: recognizer.__setattr__('input', image)

  stats = {}
  for filename in glob.glob("{}/*.txt".format(folder)):
    imageName = filename[:-4]

    def outputHandler(recognizedDigits):
      recognized = "".join(map(str, recognizedDigits))
      expected = open(filename).read()

      # limit our efforts to the 5 left digits
      if len(recognized) == 8 and len(expected) == 8:
        recognized=recognized[0:5]
        expected=expected[0:5]

        print("{}: Recognized={} Expected={} Gagne={}".format(imageName, recognized, expected, recognized == expected))
        stats[imageName] = { 'recognized': recognized, 'expected': expected }
      else:
        print("{}: Ignoring bogus - Recognized={} Expected={}".format(imageName, recognized, expected))
    recognizer.outputHandler = outputHandler
    try:
      extractor.input = imageName
    except Exception as e:
      print("{}: Error {}".format(imageName, e))

  def countSimilarDigits(a, b):
    similar = 0

    if len(a) != len(b):
      return 0

    for index, d in enumerate(a):
      if b[index] == d:
        similar = similar + 1

    return similar

  success = 0
  total = 0
  successDigits = 0
  totalDigits = 0

  for fn in stats:
    if stats[fn]['recognized'] == stats[fn]['expected']:
      success = success + 1
      successDigits = successDigits + len(stats[fn]['expected'])
      totalDigits = totalDigits + len(stats[fn]['expected'])
    else:
      successDigits = successDigits + countSimilarDigits(stats[fn]['recognized'], stats[fn]['expected'])
      totalDigits = totalDigits + len(stats[fn]['expected'])
    total = total + 1

  print("{} recognized out of {} images => {:.0%}".format(success, total, success/total))
  print("{} recognized out of {} digits => {:.0%}".format(successDigits, totalDigits, successDigits/totalDigits))

  perDigits = dict()
  for i in range(0, 10):
    perDigits["{}".format(i)] = { 'expected': 0, 'recognized': 0 }

  for fn in stats:
    expected = stats[fn]['expected']
    recognized = stats[fn]['recognized']
    if len(expected) == len(recognized):
      for i in range(0, len(expected)):
        expectedDigit = stats[fn]['expected'][i]
        foundDigit = stats[fn]['recognized'][i]

        perDigits[expectedDigit]['expected'] = perDigits[expectedDigit]['expected'] + 1
        if expectedDigit == foundDigit:
          perDigits[expectedDigit]['recognized'] = perDigits[expectedDigit]['recognized'] + 1
    else:
      print("lengths do not match {} <> {}".format(stats[fn]['recognized'], stats[fn]['expected']))

  for k in sorted(perDigits.keys()):
    print("{}: {}/{} - {:.0%}".format(k, perDigits[k]['recognized'], perDigits[k]['expected'], perDigits[k]['recognized']/perDigits[k]['expected']))

def main():
  parser = argparse.ArgumentParser()

  subparsers = parser.add_subparsers(dest='subparser_name')

  exfd = subparsers.add_parser('experiment')
  exfd.add_argument('images', nargs='+')

  xd = subparsers.add_parser('extract-digits').add_argument("image")

  subparsers.add_parser('label-samples').add_argument("folder")
  subparsers.add_parser('test-samples').add_argument("folder")
  subparsers.add_parser('train-samples').add_argument("folder")

  args = parser.parse_args()

  if args.subparser_name == 'experiment':
    experiment(args.images)
  elif args.subparser_name == 'extract-digits':
    print(extract(args.image))
  elif args.subparser_name == 'label-samples':
    labelSamples(args.folder)
  elif args.subparser_name == 'train-samples':
    trainWithSamples(args.folder)
  elif args.subparser_name == 'test-samples':
    testSamples(args.folder)
  else:
    parser.print_help()

if __name__ == '__main__':
  main()
