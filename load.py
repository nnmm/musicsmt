# coding: utf-8
import logging
import config
from music2vec import *
from sys import argv

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# the models
m = []
for i in range(1, len(argv)):
	m.append(Music2Vec.load(config.model_dir + argv[i]))
