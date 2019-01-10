# Gazmeter


## Dependencies

* Python3 + OpenCV3
  A bit hard to install on os X ... see:
  https://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/

## Notes

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

