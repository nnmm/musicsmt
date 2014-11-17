# coding: utf-8
import config
from gensim.models.word2vec import Word2Vec
from dictionary import WiktionaryDict
from registration import LeastSquaresTransform
from sys import argv, exit
import numpy

if len(argv) != 4:
	sys.exit(2)
# the models
m = []
m.append(Word2Vec.load(config.model_dir + argv[1]))
m.append(Word2Vec.load(config.model_dir + argv[2]))
fraction = float(argv[3])
wd = WiktionaryDict(config.dict_file)
transl = wd.unique_translations(m[0], m[1])
print "Translation dict ..."
train = dict(transl.items()[:int(len(transl)*fraction)])
lsq = LeastSquaresTransform(m[0], m[1], train)
transf = lsq.find_minimum()
print "Transformation ..."
numpy.save('leastsq_' + argv[1] + '_' + argv[2] + '_' + argv[3], transf)