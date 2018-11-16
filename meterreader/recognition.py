import cv2
import numpy as np
from . import VisibleStage

def hog_of_digit(image):
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

# class Learner:
#   def __init__(self):
#     C = 0
#     gamma = 0

#     svm = cv2.ml.SVM_create()
#     svm.setType(cv2.ml.SVM_C_SVC)
#     svm.setKernel(cv2.ml.SVM_RBF)
#     svm.setC(C)
#     svm.setGamma(gamma)

#     svm.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)

#     # Save trained model
#     svm->save("digits_svm_model.yml");

#     # Test on a held out test set
#     testResponse = svm.predict(testData)[1].ravel()

'''
input: a list of image
output: a list of digit, as recognized
'''
class DigitsRecognizer(VisibleStage):
  svm = cv2.ml.SVM_load('svm_data.dat')

  def _initWindows(self):
    cv2.namedWindow('DigitsRecognizer')
    cv2.createTrackbar('blur', 'DigitsRecognizer', self._params['blur'], 5, self.guicb)
    cv2.createTrackbar('thrs1', 'DigitsRecognizer', self._params['thrs1'], 255, self.guicb)
    cv2.createTrackbar('thrs2', 'DigitsRecognizer', self._params['thrs2'], 255, self.guicb)

  def guicb(self, value):
    print(repr(value))
    params = self._params
    params['blur'] = cv2.getTrackbarPos('blur', 'DigitsRecognizer')
    if params['blur'] != 0 and params['blur'] % 2 != 1:
      params['blur'] = params['blur'] + 1
      cv2.setTrackbarPos('blur', 'DigitsRecognizer', params['blur'])

    params['thrs1'] = cv2.getTrackbarPos('thrs1', 'DigitsRecognizer')
    params['thrs2'] = cv2.getTrackbarPos('thrs2', 'DigitsRecognizer')
    self.params = params

  def _process(self):
    output = []
    if self._showYourWork:
      (self.debugDigitHeight, self.debugDigitWidth) = self.input[0].shape[:2]
      # 20px + image + 20px margin + image + 20px margin + ...
      self.debugWidth = 20 + len(self.input) * (self.debugDigitWidth + 20)
      # 20px top margin + image + 20px margin  (3 times for three lines in the output)
      self.debugHeight = 20 + 3 * (self.debugDigitHeight + 20)
      self.debugImage = np.zeros((self.debugHeight, self.debugWidth, 3), dtype=np.uint8)

    for index, image in enumerate(self.input):
      output.append(self.process_digit(image, index))

    if self._showYourWork:
      cv2.imshow('DigitsRecognizer', cv2.resize(self.debugImage, (self.debugWidth*3, self.debugHeight*3)))

    self.outputHandler(output)

  def process_digit(self, image, index):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if self.params['blur'] > 0:
      blur = cv2.GaussianBlur(grayImage,(self.params['blur'],self.params['blur']),0)
    else:
      blur = grayImage
    _, threshold = cv2.threshold(blur,self.params['thrs1'],self.params['thrs2'],cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imageHash = hog_of_digit(threshold)
    detected = 0
    if len(imageHash) > 0:
      detected = self.svm.predict(np.float32([imageHash]))
      detected = np.int(detected[1].ravel()[0])

    if self._showYourWork:
      self.draw_debug_image(index, 0, image)
      self.draw_debug_image(index, 1, blur)
      self.draw_debug_image(index, 2, threshold)
      self.draw_result(index, detected)

    return detected

  def draw_debug_image(self, indexX, indexY, image):
    if len(image.shape) != 3:
      image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    x = 20+indexX*(self.debugDigitWidth+20)
    y = 20+indexY*(self.debugDigitHeight+20)

    self.debugImage[y:y+image.shape[0], x:x+image.shape[1]] = image

  def draw_result(self, indexX, result):
    x = 20+indexX*(self.debugDigitWidth+20)
    y = 3*(self.debugDigitHeight+20) + 10
    cv2.putText(self.debugImage, str(result), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

