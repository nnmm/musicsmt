# coding: utf-8
import config
from gensim.models.word2vec import Word2Vec
from dictionary import WiktionaryDict
from registration import LeastSquaresTransform
from sys import argv, exit

if len(argv) != 4:
	sys.exit(2)
# the models
m = []
m.append(Word2Vec.load(config.model_dir + argv[1]))
m.append(Word2Vec.load(config.model_dir + argv[2]))
wd = WiktionaryDict(config.dict_file)
transl = wd.unique_translations(m[0], m[1])
print "Translation dict ..."
train = dict(transl.items()[len(transl)/4:])
test = dict(transl.items()[:len(transl)/4])
lsq = LeastSquaresTransform(m[0], m[1], train)
transf = lsq.find_minimum()
m[0].syn0norm = transf.dot(m[0].syn0norm.T).T
print "Transformation ..."

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
print correct
print incorrect