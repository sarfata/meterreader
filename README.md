# Gazmeter

## Installation

### MacOS

    brew install python3
    brew postinstall python3 (for pip)
    pip3 install -r requirements.txt (will install opencv-python prebuilt binaries)
    python3 / import cv2 / cv2.__version__ => 4.1.1

## Using

Get images from Amazon:

    aws s3 cp s3://metering-vanves/img img --recursive --exclude '*' --include 'image-2019*-0900*jpg'

Experiment:

    ./meterreader.py experiment img201911/*jpg

Label some images:

    python3 meterreader.py label-samples img

For each image shown, click on the image window and type the digit you see. Press enter when done. This will add a txt file next to each image with the value.

Test recognition:

    python3 meterreader.py test-samples img

Process an entire folder of images:

    ./meterreader.py process-images img201911

## Good resources

### Installing OpenCV (2019 new mac Edition)

### KNN

- https://towardsdatascience.com/scanned-digits-recognition-using-k-nearest-neighbor-k-nn-d1a1528f0dea
- https://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python

## Notes

### Version of Jan 11 2019

SVM recognition:

46 recognized out of 79 images => 58%
516 recognized out of 553 digits => 93%
0: 25/26 - 96%
1: 183/183 - 100%
2: 22/36 - 61%
3: 31/33 - 94%
4: 32/34 - 94%
5: 86/88 - 98%
6: 49/50 - 98%
7: 19/26 - 73%
8: 35/42 - 83%
9: 34/35 - 97%

KNN recognition:

26 recognized out of 78 images => 33%
467 recognized out of 546 digits => 86%
0: 22/26 - 85%
1: 178/181 - 98%
2: 22/36 - 61%
3: 21/33 - 64%
4: 32/34 - 94%
5: 77/86 - 90%
6: 37/50 - 74%
7: 18/26 - 69%
8: 27/40 - 68%
9: 33/34 - 97%

### Version of Jan 9 2019

    ./meterreader.py train-samples img
    0: 36 - 6%
    1: 190 - 30%
    2: 46 - 7%
    3: 40 - 6%
    4: 43 - 7%
    5: 94 - 15%
    6: 56 - 9%
    7: 29 - 5%
    8: 54 - 8%
    9: 52 - 8%
    SVM Model saved!

    # Testing with only 6 digits.

    ./meterreader.py test-samples img
    71 recognized out of 83 images => 86%
    389 recognized out of 415 digits => 94%
    0: 13/14 - 93%
    1: 176/183 - 96%
    2: 17/18 - 94%
    3: 13/15 - 87%
    4: 19/20 - 95%
    5: 73/79 - 92%
    6: 37/39 - 95%
    7: 9/11 - 82%
    8: 16/19 - 84%
    9: 16/17 - 94%

### Version of Nov 17 2018

    ./meterreader.py train-samples img/
    0: 21 - 5%
    1: 121 - 29%
    2: 28 - 7%
    3: 29 - 7%
    4: 26 - 6%
    5: 60 - 14%
    6: 39 - 9%
    7: 22 - 5%
    8: 31 - 7%
    9: 39 - 9%
    SVM Model saved!

./meterreader.py test-samples img
13 recognized out of 54 images => 24%
357 recognized out of 424 digits => 84%
