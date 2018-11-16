'''
Defines a transformation in the image recognition pipeline.

  - Processing is defined by a set of parameters (params)
    and an input (input).
  - Every time one of them changes, the outputHandler will be called with the
    result of the internal processing (_process).
'''
class Stage(object):
  _params = {}
  _input = None
  outputHandler = staticmethod(lambda x: x)

  def __init__(self, params):
    self._params = params

  @property
  def input(self):
    return self._input

  @input.setter
  def input(self, value):
    self._input = value
    self._process()

  @property
  def params(self):
    return self._params

  @params.setter
  def params(self, value):
    self._params = value
    self._process()

  '''
  Process input and call outputHandler with the output.
  '''
  def _process(self):
    self.outputHandler(self.input)


'''
A stage that creates window to show work and manipulate parametes directly.
'''
class VisibleStage(Stage):
  def __init__(self, params, showYourWork = False):
    super().__init__(params)
    self._showYourWork = showYourWork
    if (showYourWork):
      self._initWindows()

  def _initWindows(self):
    pass
