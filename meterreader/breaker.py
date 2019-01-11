import cv2
from . import VisibleStage

'''
Split an image into smaller images.

input: A landscape image with a series of digits
output: A list of images, one per digit
'''
class DigitsBreaker(VisibleStage):
  def _process(self):
    digitsImage = self.input

    if digitsImage is None:
      self.outputHandler([])
      return

    (h, w) = digitsImage.shape[:2]
    digits = []
    for i in range(8):
      x1 = i * int(w / 8)
      x2 = ((i+1) * int(w/8)) - 1
      digit = digitsImage[0:h, x1:x2]
      digits.append(digit)

    if self._showYourWork:
      current_stack_y = 50
      for i in range(8):
        wName = 'digits{}'.format(i)
        cv2.namedWindow(wName)
        (h, w) = digits[i].shape[:2]
        cv2.imshow(wName, cv2.resize(digits[i], (w*4, h*4)))
        cv2.moveWindow(wName, 20, current_stack_y)
        current_stack_y += 60

    self.outputHandler(digits)