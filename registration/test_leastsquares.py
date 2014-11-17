import unittest
from least_squares_transform import LeastSquaresTransform
import numpy
import scipy
import cProfile
import pstats

class LeastSquaresTest(unittest.TestCase):
	"""Testing suite for Kernel Correlation"""

	def setUp(self):
		print "==== " + self.id().split('.')[-1] + " ===="
		self.pr = cProfile.Profile()
		self.pr.enable()

		self.mdim = 4
		self.sdim = 4
		self.num_vectors = 100

		self.correct_transform = numpy.random.rand(self.sdim, self.mdim)
		model = self.normed_rand(self.num_vectors, self.mdim)
		scene = self.correct_transform.dot(model.T).T
		self.reg = KernelCorrelation(model, scene)


	def tearDown(self):
		print "\n"


	def print_profile(self):
		with open('kcstats', 'w') as stream:
			p = pstats.Stats(self.pr, stream=stream)
			p.strip_dirs()
			p.sort_stats('cumtime')
			p.print_stats()

	# ---------------------------------------------

	def test_percentage_correct(self):
		correct, incorrect = 0, 0
		for source_word in m[0].vocab.keys():
			if source_word in test:
				result = wd.check(source_word, m, n=int(argv[3]))
				if result is not None:
					print str(correct) + "/" +  str(incorrect)
					if result:
						correct = correct + 1
					else:
						incorrect = incorrect + 1

if __name__ == '__main__':
	numpy.load('../leastsq_enwiki300M_frwiki300M_1.npy')
	unittest.main()