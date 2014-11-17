# coding: utf-8
import numpy
import scipy.linalg
from read_ding import WiktionaryDict
from registration import Registration


class LeastSquaresTransform(Registration):
	"""Compute the optimal transformation matrix given two set of vectors that correspond"""
	def __init__(self, model, scene, word2vec_dict=None):
		if word2vec_dict is not None:
			model, scene = self.from_word2vec(model, scene, word2vec_dict)
		super(LeastSquaresTransform, self).__init__(model, scene)


	def from_word2vec(self, model, scene, word2vec_dict):
		ordered_model = numpy.empty((len(word2vec_dict), model.syn0norm.shape[1]))
		ordered_scene = numpy.empty((len(word2vec_dict), scene.syn0norm.shape[1]))
		ind = 0
		for k, v in word2vec_dict.iteritems():
			ordered_model[ind] = model.syn0norm[model.vocab[k].index]
			ordered_scene[ind] = scene.syn0norm[scene.vocab[v].index]
			ind = ind + 1
		return ordered_model, ordered_scene


	def find_minimum(self):
		W = numpy.empty(self.transform_shape)
		for ii in xrange(0,self.transform_shape[0]):
			W[ii, :] = scipy.linalg.lstsq(self.model, self.scene[:,ii])[0]
		return W

