# coding: utf-8
# import modules and set up logging
import logging
import config
from music2vec import *
from sys import argv

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# -----------------------------------------------------------------------------------------

for i in range(1, len(argv)):
	sentences = LineSentence(config.data_dir + argv[i] + '.txt')
	m = Music2Vec(sentences, size=2, workers=4, min_count=1)
	m.init_sims(replace=True)
	m.save(config.model_dir + argv[i], ignore=[], separately=[])
