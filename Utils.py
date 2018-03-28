#!/usr/bin/python

from nltk import RegexpTokenizer as RT
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class Utils():
	def __init__(self):
		# create array of selected stop words and punctuation
		self.stop_words = set(stopwords.words('english'))
		self.stop_words.add(',')
		self.stop_words.add('.')
		self.stop_words.add('?')
		self.stop_words.add(';')
		self.stop_words.add(':')
		self.stop_words.add('$')

	def regex_tokenizer(self, sent):
		regex_tokenizer = RT('\w+|\[M:.*?\]|\S+')
		tokens = regex_tokenizer.tokenize(sent)

		return [t for t in tokens if not t in self.stop_words and len(t) > 2]
