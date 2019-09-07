#!/usr/bin/env python3

import cv2
import argparse
import datetime
import math
import numpy as np
import imutils
import sys
import os.path
import glob
import random
import traceback
import re
from meterreader import DigitalCounterExtraction, DigitsBreaker, DigitsCleaner, DigitsRecognizer, DigitRecognition, MakeHorizontal

PARAMS = {
    'horizontality': {

    },
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
        'blur': 0,
        'thrs1': 90,
        'thrs2': 255
    },
    'recognizer': {

    }
}


def experiment(images):
    makeHorizontal = MakeHorizontal(PARAMS['horizontality'], False)
    extractor = DigitalCounterExtraction(PARAMS['extraction'], True)
    breaker = DigitsBreaker(PARAMS['breaker'], False)
    cleaner = DigitsCleaner(PARAMS['cleaner'], True)
    recognizer = DigitsRecognizer(PARAMS['recognizer'])

    makeHorizontal.outputHandler = lambda image: extractor.__setattr__(
        'input', image)
    extractor.outputHandler = lambda image: breaker.__setattr__('input', image)
    breaker.outputHandler = lambda image: cleaner.__setattr__('input', image)
    cleaner.outputHandler = lambda digits: recognizer.__setattr__(
        'input', digits)
    recognizer.outputHandler = lambda digits: print(
        "Reco output: {}".format(repr(digits)))

    imageIndex = 0
    makeHorizontal.input = cv2.imread(images[imageIndex])
    while (True):
        print("waiting for key")
        key = cv2.waitKey(0)
        print("key={}".format(key))

        if key == 27 or key == ord('q'):
            break
        elif key == ord('p') or key == 2:
            imageIndex = (imageIndex - 1) % len(images)
            makeHorizontal.input = cv2.imread(images[imageIndex])
        elif key == ord('n') or key == 3:
            imageIndex = (imageIndex + 1) % len(images)
            makeHorizontal.input = cv2.imread(images[imageIndex])
    else:
        print("Key pressed: {}".format(key))


def extract(image):
    makeHorizontal = MakeHorizontal(PARAMS['horizontality'], False)
    extractor = DigitalCounterExtraction(PARAMS['extraction'], True)
    breaker = DigitsBreaker(PARAMS['breaker'], False)
    cleaner = DigitsCleaner(PARAMS['cleaner'], True)
    recognizer = DigitsRecognizer(PARAMS['recognizer'])

    output = {'value': None}

    def outputHandler(v):
        output['value'] = "".join(map(str, v))

    makeHorizontal.outputHandler = lambda image: extractor.__setattr__(
        'input', image)
    extractor.outputHandler = lambda image: breaker.__setattr__('input', image)
    breaker.outputHandler = lambda image: cleaner.__setattr__('input', image)
    cleaner.outputHandler = lambda image: recognizer.__setattr__(
        'input', image)
    recognizer.outputHandler = outputHandler

    makeHorizontal.input = cv2.imread(image)
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

    makeHorizontal = MakeHorizontal(PARAMS['horizontality'], False)
    extractor = DigitalCounterExtraction(PARAMS['extraction'], False)
    breaker = DigitsBreaker(PARAMS['breaker'], False)
    cleaner = DigitsCleaner(PARAMS['cleaner'], False)
    recognition = DigitRecognition()

    makeHorizontal.outputHandler = lambda image: extractor.__setattr__(
        'input', image)
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
            print("{}: Skipping because of invalid expected results: {}".format(
                filename, result))
            continue

        def outputHandler(digitsImages):
            for image in digitsImages:
                recognition.train(image, result.pop(0))

        cleaner.outputHandler = outputHandler
        try:
            makeHorizontal.input = cv2.imread(filename)
        except Exception as e:
            print("{}: Error {}".format(filename, e))
            traceback.print_tb(e.__traceback__)

    recognition.saveModel()


def testSamples(folder):
    makeHorizontal = MakeHorizontal(PARAMS['horizontality'], False)
    extractor = DigitalCounterExtraction(PARAMS['extraction'], False)
    breaker = DigitsBreaker(PARAMS['breaker'], False)
    cleaner = DigitsCleaner(PARAMS['cleaner'], False)
    recognizer = DigitsRecognizer(PARAMS['recognizer'])

    makeHorizontal.outputHandler = lambda image: extractor.__setattr__(
        'input', image)
    extractor.outputHandler = lambda image: breaker.__setattr__('input', image)
    breaker.outputHandler = lambda image: cleaner.__setattr__('input', image)
    cleaner.outputHandler = lambda image: recognizer.__setattr__(
        'input', image)

    stats = {}
    for filename in glob.glob("{}/*.txt".format(folder)):
        imageName = filename[:-4]

        def outputHandler(recognizedDigits):
            recognized = "".join(map(str, recognizedDigits))
            expected = open(filename).read()

            if len(recognized) == 8 and len(expected) == 8:
                # limit our efforts to 7 digits because the last one is hard.
                recognized = recognized[0:7]
                expected = expected[0:7]

                print("{}: Recognized={} Expected={} Gagne={}".format(
                    imageName, recognized, expected, recognized == expected))
                stats[imageName] = {
                    'recognized': recognized, 'expected': expected}
            else:
                print(
                    "{}: Ignoring bogus - Recognized={} Expected={}".format(imageName, recognized, expected))
        recognizer.outputHandler = outputHandler
        try:
            makeHorizontal.input = cv2.imread(imageName)
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
            successDigits = successDigits + \
                countSimilarDigits(
                    stats[fn]['recognized'], stats[fn]['expected'])
            totalDigits = totalDigits + len(stats[fn]['expected'])
        total = total + 1

    print("{} recognized out of {} images => {:.0%}".format(
        success, total, success/total))
    print("{} recognized out of {} digits => {:.0%}".format(
        successDigits, totalDigits, successDigits/totalDigits))

    perDigits = dict()
    for i in range(0, 10):
        perDigits["{}".format(i)] = {'expected': 0, 'recognized': 0}

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
            print("lengths do not match {} <> {}".format(
                stats[fn]['recognized'], stats[fn]['expected']), file=sys.stderr)

    for k in sorted(perDigits.keys()):
        print("{}: {}/{} - {:.0%}".format(k, perDigits[k]['recognized'], perDigits[k]
                                          ['expected'], perDigits[k]['recognized']/perDigits[k]['expected']))


def processImages(folder):
    results = dict()
    for filename in glob.glob("{}/*.jpg".format(folder)):
        match = re.search(r"image-(.*).jpg", filename)
        if (match):
            date = match.group(1)
            try:
                parsed_date = datetime.datetime.strptime(date, "%Y%m%d-%H%M%S")
                date = parsed_date.strftime("%Y%m%d %H%M%S")
            except Exception as e:
                print("{}: date error {}".format(filename, e), file=sys.stderr)
                pass

            try:
                results[date] = extract(filename)
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception as e:
                print("{}: {}".format(filename, e), file=sys.stderr)
        else:
            print("{}: Unrecognized filename", filename, file=sys.stderr)

    for date in sorted(results.keys()):
        print("{},{}".format(date, results[date]))


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='subparser_name')

    exfd = subparsers.add_parser('experiment')
    exfd.add_argument('images', nargs='+')

    xd = subparsers.add_parser(
        'extract-digits').add_argument("images", nargs='+')

    subparsers.add_parser('label-samples').add_argument("folder")
    subparsers.add_parser('test-samples').add_argument("folder")
    subparsers.add_parser('train-samples').add_argument("folder")
    subparsers.add_parser('process-images').add_argument("folder")

    args = parser.parse_args()

    if args.subparser_name == 'experiment':
        experiment(args.images)
    elif args.subparser_name == 'extract-digits':
        for image in args.images:
            result = None
            try:
                result = extract(image)
                print("{}: {}".format(image, result))
            except KeyboardInterrupt as kbe:
                sys.exit(0)
            except:
                print("{}: Error {}".format(image, sys.exc_info()))
    elif args.subparser_name == 'process-images':
        processImages(args.folder)
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
