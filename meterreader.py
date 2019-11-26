#!/usr/bin/env python3

import cv2
import argparse
import datetime
import math
import numpy as np
import imutils
import sys
import os.path
import os
import glob
import random
import traceback
from meterreader import DigitalCounterExtraction, DigitsBreaker, DigitsCleaner, DigitsRecognizer, DigitRecognition, MakeHorizontal, parse_filename_date
from influxdb import InfluxDBClient

PARAMS = {
    'horizontality': {

    },
    'extraction': {
        'thrs1': 143,
        'thrs2': 50,
        'blur': 5,
        'epsilon': 10
    },
    'breaker': {
        'numdigits': 8
    },
    'cleaner': {
        'blur': 0,
        'thrs1': 59,
        'thrs2': 130
    },
    'recognizer': {

    }
}


def experiment(images):
    makeHorizontal = MakeHorizontal(PARAMS['horizontality'], True)
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

    def select_image(idx):
        try:
            print("-- {}".format(images[idx]))
            image = cv2.imread(images[idx])
            if image is None:
                raise Exception("Unable to read image.")
            makeHorizontal.input = image
        except Exception as e:
            print("{}: {}".format(images[idx], e), file=sys.stderr)

    imageIndex = 0
    select_image(imageIndex)
    while (True):
        print("waiting for key")
        key = cv2.waitKey(0)
        print("key={}".format(key))

        if key == 27 or key == ord('q'):
            break
        elif key == ord('p') or key == 2:
            imageIndex = (imageIndex - 1) % len(images)
            select_image(imageIndex)
        elif key == ord('n') or key == 3:
            imageIndex = (imageIndex + 1) % len(images)
            select_image(imageIndex)
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
        date = parse_filename_date(filename)
        if date:
            date = date.strftime("%Y%m%d %H%M%S")

            try:
                results[date] = extract(filename)
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception as e:
                print("{}: {}".format(filename, e), file=sys.stderr)
        else:
            print("{}: Unrecognized filename", filename, file=sys.stderr)

    last = None
    nonvalid = 0
    bogus = 0

    for date in sorted(results.keys()):
        valid = True
        value = None
        try:
            value = int(results[date])
        except ValueError:
            nonvalid = nonvalid + 1
            continue

        if last is not None:
            if value < last:
                valid = False
            elif value > last + 10000:
                valid = False

        print("{},{},{}".format(date, value, valid))

        if not valid:
            bogus = bogus + 1
        else:
            last = value

    print("{} invalid values and {} bogus values out of {}".format(
        nonvalid, bogus, len(results.keys())))


def extractAndUpload(images):
    print("connecting to {}".format(os.getenv('INFLUX_HOST')))
    client = InfluxDBClient(host=os.getenv('INFLUX_HOST'), port=443, ssl=True, username=os.getenv('INFLUX_USERNAME'), database=os.getenv('INFLUX_DATABASE'),
                            password=os.getenv('INFLUX_PASSWORD'), path='/influxdb')
    data = []

    for filename in images:
        try:
            digits = extract(filename)

            value = int(digits)
            date = parse_filename_date(filename)
            if date:
                date = date.isoformat()
                print("{}, {}".format(date, value))
                data.append({
                    "measurement": "gazmeter",
                    "time": date,
                    "fields": {
                        "value": value
                    }
                })
            else:
                print("Invalid date: {}".format(filename))
        except Exception as e:
            print("Error {}: {}".format(filename, e))

    print("Pushing data to influxdb!")
    client.write_points(data)


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

    subparsers.add_parser('upload').add_argument("images", nargs='+')

    args = parser.parse_args()

    if args.subparser_name == 'experiment':
        experiment(args.images)
    elif args.subparser_name == 'extract-digits':
        for image in args.images:
            result = None
            try:
                result = extract(image)
                print("{}: {}".format(image, result))
            except KeyboardInterrupt:
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
    elif args.subparser_name == 'upload':
        extractAndUpload(args.images)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
