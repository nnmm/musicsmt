# coding: utf-8
import config
from read_ding import WiktionaryDict
from registration import LeastSquaresTransform
from gensim.models.word2vec import Word2Vec
# the models
m = []
m.append(Word2Vec.load(config.model_dir + 'micro_900_900_600_rkc'))
m.append(Word2Vec.load(config.model_dir + 'micro_600_750_900'))

correct, incorrect = 0, 0
for source_word in m[0].vocab.keys():
	target_word = m[1].most_similar([m[0].syn0norm[m[0].vocab[source_word].index]], topn=1)[0][0]
	print source_word + " - " + target_word
	if source_word == target_word:
		correct = correct + 1
	else:
		incorrect = incorrect + 1
print correct
print incorrect