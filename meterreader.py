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
from meterreader import DigitalCounterExtraction, DigitsBreaker, DigitsRecognizer, hog_of_digit

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

def extract(image):
  extractor = DigitalCounterExtraction(PARAMS['extraction'], True)
  breaker = DigitsBreaker(PARAMS['breaker'], False)
  reader = DigitsRecognizer(PARAMS['recognizer'], True)

  output = { 'value': None }
  def outputHandler(v):
    output['value'] = "".join(map(str, v))

  extractor.outputHandler = lambda image: breaker.__setattr__('input', image)
  breaker.outputHandler = lambda image: reader.__setattr__('input', image)
  reader.outputHandler = outputHandler

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
  extractor.outputHandler = lambda image: breaker.__setattr__('input', image)

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
        hog = hog_of_digit(image)
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

    breaker.outputHandler = outputHandler
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
  recognizer = DigitsRecognizer(PARAMS['recognizer'], False)

  extractor.outputHandler = lambda image: breaker.__setattr__('input', image)
  breaker.outputHandler = lambda image: recognizer.__setattr__('input', image)

  stats = {}
  for filename in glob.glob("{}/*.txt".format(folder)):
    imageName = filename[:-4]

    def outputHandler(recognizedDigits):
      recognized = "".join(map(str, recognizedDigits))
      expected = open(filename).read()
      print("{}: Recognized={} Expected={} Gagne={}".format(imageName, recognized, expected, recognized == expected))
      stats[imageName] = { 'recognized': recognized, 'expected': expected }
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




# Old training method: digit by digit. Now we use the whole image and a manual
# reading of the image. Much easier and can be used for testing as well.

# def train(images):
#   extractor = DigitalCounterExtraction(PARAMS['extraction'], False)
#   breaker = DigitsBreaker(PARAMS['breaker'], False)
#   extractor.outputHandler = lambda image: breaker.__setattr__('input', image)

#   for image in images:
#     def digit_processor(digits):
#       for index,d in enumerate(digits):
#         ask_and_save(d, index)
#     def ask_and_save(d, index):
#       (w,h) = d.shape[:2]
#       cv2.imshow('digit', cv2.resize(d, (h*8, w*8)))
#       key = cv2.waitKey(0)

#       if (key >= ord('0') and key <= ord('9')) or key == ord(' '):
#         if key == ord(' '):
#           key = ord('x')

#         fname = 'trainingdata/{}-{}_{}.png'.format(os.path.basename(image), index, chr(key))
#         print("Saving {} -> {}".format(chr(key), fname))
#         cv2.imwrite(fname, d)
#       elif key == ord('q'):
#         sys.exit(0)

#     print("Processing: {}".format(image))
#     breaker.outputHandler = digit_processor

#     # This triggers the processing, the callbacks, etc.
#     extractor.input = image

# def learn():
#   counter = {}

#   hogdata = []
#   results = []

#   for filename in glob.glob('trainingdata/*png'):
#     digit = filename.split('_')[1][0]
#     image = cv2.imread(filename)
#     hog = hog_of_digit(image)
#     if not digit in counter:
#       counter[digit] = 0
#     counter[digit] = counter[digit] + 1
#     if len(hog) > 0:
#       print("{}: {}=>{}".format(filename, repr(hog.shape), digit))
#       if digit != 'x':
#         hogdata.append(hog)
#         results.append(digit)
#     else:
#       print("{}: error getting hog".format(filename))

#   total = sum(counter.values())
#   for k in sorted(counter.keys()):
#     print("{}: {} - {:.0%}".format(k, counter[k], counter[k]/total))

#   svm = cv2.ml.SVM_create()
#   svm.setKernel(cv2.ml.SVM_RBF)
#   # svm.setType(cv2.ml.SVM_C_SVC)

#   trainingData = np.float32(hogdata).reshape(-1,81)
#   trainingResult = np.asarray(results, dtype=int)[:,np.newaxis]

#   svm.trainAuto(trainingData, cv2.ml.ROW_SAMPLE, trainingResult)
#   svm.save('svm_data.dat')
#   print("SVM Model saved!")

