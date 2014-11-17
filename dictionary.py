import re
from collections import OrderedDict

class WiktionaryDict(object):
	"""Construct a dictionary from two Wiktionary ding files, downloaded from en.wiktionary.org/wiki/User:Matthias_Buchmeier/download"""
	def __init__(self, file_path):
		super(WiktionaryDict, self).__init__()
		with open(file_path) as f:
			self.entries = list(f)[6:]
		self.build_dict()


	def build_dict(self):
		self._dictionary = {}
		self.re_brackets = re.compile(u'/[^/]+/ ?|\[[^\]]+\] ?|\([^)]+\) ?', re.UNICODE)
		self.re_splitter = re.compile(u',|;|\{[^}]+\} ?', re.UNICODE)

		for line in self.entries:
			both = line.split(' :: ')
			if len(both) != 2:
				continue
			#if len(both) != 0 and both[0][:4] == 'calf':

			source = self.extract_translations(both[0])
			target = self.extract_translations(both[1])
			if len(source) != 0 and len(target) != 0:
				if source[0] in self._dictionary:
					self._dictionary[source[0]] = self._dictionary[source[0]].union(set(target))
				else:
					self._dictionary[source[0]] = set(target)


	def get_pos(self):
		self.re_pos = re.compile(u'', re.UNICODE)
		pos_dict = {}
		for line in self.entries:
			both = line.split(' :: ')
			if len(both) != 2:
				continue
			lemma = self.extract_translations(both[0])
			if len(lemma) != 1:
				continue
			word = lemma[0]
			pos = None
			if '{v' in both[0]:
				pos = 'v'
			elif '{m' in both[0] or '{n' in both[0] or '{f' in both[0]:
				pos = 'n'
			elif '{adj}' in both[0]:
				pos = 'adj'
			elif '{adv}' in both[0]:
				pos = 'adv'
			if pos is None:
				continue
			elif pos in pos_dict:
				pos_dict[pos].append(word)
			else:
				pos_dict[pos] = [word]
		for k in pos_dict.keys():
			pos_dict[k] = list(OrderedDict.fromkeys(pos_dict[k]))
		return pos_dict
				

	def extract_translations(self, raw_string, debug=False):
		# discard everything inside () or []
		without_brackets = self.re_brackets.sub('', raw_string).strip()
		# this can result in artifacts when there are nested brackets, but this is not so important
		split_by_words = [res.strip() for res in self.re_splitter.split(without_brackets)]
		# no phrases
		single_words = [w for w in split_by_words if len(w.split()) == 1]
		if debug:
			print 'DEBUGGING ' + raw_string
			print 'without_brackets: ' + without_brackets
			print 'split_by_words: ' + str(split_by_words)
			print 'single_words: ' + str(single_words)
		return single_words


	def check(self, source_word, m, n=1):
		if source_word in self.unique_shared_translations:
			target_words = zip(*m[1].most_similar([m[0].syn0norm[m[0].vocab[source_word].index]], topn=n))[0]
			print source_word + " - " + str(self._dictionary[source_word]) + " - " + str(target_words)
			for x in range(n):
				if target_words[x] in self._dictionary[source_word]:
					return True
			return False


	def unique_translations(self, source, target):
		"""Create a mapping between vocabularies of word2vec objects source and target"""
		shared_keys = set(source.vocab.keys()).intersection(set(self._dictionary.keys()))
		unique_target_vocab = set([])
		multiples = set([])
		for v in self._dictionary.values():
			for word in v:
				if word in unique_target_vocab:
					multiples.add(word)
				unique_target_vocab.add(word)
		unique_target_vocab.difference_update(multiples)

		shared_values = unique_target_vocab.intersection(set(target.vocab.keys()))
		# unpack the values from the set
		unique_translations = {k : next(iter(self._dictionary[k])) for k in shared_keys
			if len(self._dictionary[k]) == 1}
		unique_shared_translations = {k: v for k, v in unique_translations.iteritems() if v in shared_values}
		self.unique_shared_translations = unique_shared_translations
		return unique_shared_translations