import cv2
import numpy as np
import os.path
from .stage import Stage


class DigitRecognition(object):
  svm = None
  learningStats = dict()
  hogdata = list()
  results = list()

  def __init__(self):
    if os.path.exists('svm_data.dat'):
      self.svm = cv2.ml.SVM_load('svm_data.dat')
    else:
      print("Warning! No SVM model found. Train the model!!")

  def recognize(self, image):
    imageHash = self.hog_of_digit(image)

    # do the detection so we can draw it
    detected = None
    if len(imageHash) > 0 and self.svm is not None:
      detected = self.svm.predict(np.float32([imageHash]))
      detected = np.int(detected[1].ravel()[0])
    else:
      print("len(imageHash)={}  self.svm={}".format(len(imageHash), self.svm))
    return detected

  def train(self, image, expected):
    hog = self.hog_of_digit(image)
    if len(hog) == 81:
      self.hogdata.append(hog)
      self.results.append(expected)
      if expected not in self.learningStats:
        self.learningStats[expected] = 0
      self.learningStats[expected] = self.learningStats[expected] + 1
    else:
      cv2.imwrite('last-digit-error.png', image)
      raise Exception("error - invalid HOG - len={} - see last-digit-error.png".format(len(hog)))

  def saveModel(self):
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_RBF)
    trainingData = np.float32(self.hogdata).reshape(-1,81)
    trainingResult = np.asarray(self.results, dtype=int)[:,np.newaxis]
    svm.trainAuto(trainingData, cv2.ml.ROW_SAMPLE, trainingResult)
    svm.save('svm_data.dat')

    total = sum(self.learningStats.values())
    for k in sorted(self.learningStats.keys()):
      print("{}: {} - {:.0%}".format(k, self.learningStats[k], self.learningStats[k]/total))

  def hog_of_digit(self, image):
    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    useSignedGradients = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels,useSignedGradients)
    # hog = cv2.HOGDescriptor()
    descriptor = hog.compute(image)
    return descriptor

class DigitsRecognizer(Stage):
  recognizer = DigitRecognition()

  def _process(self):
    output = []
    for image in self.input:
      output.append(self.recognizer.recognize(image))
    self.outputHandler(output)