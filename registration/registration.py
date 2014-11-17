class Registration(object):
	"""Base class for Point set registration methods KernelCorrelation and LeastSquaresMinimization"""
	def __init__(self, model, scene):
		super(Registration, self).__init__()
		self.model = model
		self.scene = scene
		self.transform_shape = (self.scene.shape[1], self.model.shape[1])