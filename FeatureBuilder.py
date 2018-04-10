#!/usr/bin/python

import sys, re
import math, numpy
import spacy
#from nltk.tokenize import word_tokenize
from nltk.util import ngrams

class FeatureBuilder():
	def __init__(self, mec):
		self.mec = mec
		
		self.train_acc = []
		self.test_acc = []
		self.precision = []
		self.recall = []

		self.all_problems = self.mec.alg_problems + self.mec.arith_problems + self.mec.geo_problems

	# add all word tokens into a set and sort tokens alphabetically
	def get_vocabulary(self, use_bigrams, use_trigrams):
		sys.stderr.write("Compiling vocabulary.\n")

#		vocab = [w for p in all_problems for w in word_tokenize(p[0].lower()) if (not w in self.stop_words) and (len(w) > 2)]
		vocab = [w for p in self.all_problems for w in self.mec.util.regex_tokenizer(p[0].lower())]

		if use_bigrams:
			vocab += [b for p in self.all_problems for b in ngrams(self.mec.util.regex_tokenizer(p[0].lower()), 2)]

		if use_trigrams:
			vocab += [t for p in self.all_problems for t in ngrams(self.mec.util.regex_tokenizer(p[0].lower()), 3)]

		vocab = set(vocab)
		vocab = list(vocab)
		vocab.sort()

		return vocab

	# abstract for get_weights
	def get_init_weights(self, V, use_bigrams, use_trigrams):
		sys.stderr.write("Initializing weights.\n")

		wts = numpy.zeros((3, len(V)))

		# process algebra problem set
		wts = self.get_weights(
			self.mec.alg_train_set, self.mec.alg,
			V, wts, 0, 
			use_bigrams, use_trigrams
		)

		# process arithmetic problem set
		wts = self.get_weights(
			self.mec.arith_train_set, self.mec.arith,
			V, wts, 1, 
			use_bigrams, use_trigrams
		)

		# process geometry problem set
		wts = self.get_weights(
			self.mec.geo_train_set, self.mec.geo,
			V, wts, 2, 
			use_bigrams, use_trigrams
		)

		return wts
		
	# set initial weights to raw word count probabilities
	def get_weights(self, t_set, probs, V, wts, idx, b, t):
		for p, c in t_set:
#			tokens = word_tokenize(p.lower())
			tokens = self.mec.util.regex_tokenizer(p.lower())

			for w in tokens:
				if w in V: wts[idx][V.index(w)] = probs[w]

			if b:
				bigrams = ngrams(tokens, 2)

				for b in bigrams:
					if b in V: wts[idx][V.index(b)] = probs[b]

			if t:
				trigrams = ngrams(tokens, 3)

				for t in trigrams:
					if t in V: wts[idx][V.index(t)] = probs[t]

		return wts	

	# abstract for get_features
	def get_train_features(self, V, use_bigrams, use_trigrams):
		sys.stderr.write("\nVectorizing training features.\n")

		return self.get_word_features(self.mec.train_problems, V, use_bigrams, use_trigrams)

	# abstract for get_features
	def get_test_features(self, V, use_bigrams, use_trigrams):
		sys.stderr.write("\nVectorizing test features.\n")

		return self.get_word_features(self.mec.test_problems, V, use_bigrams, use_trigrams)

	# convert word tokens into 0-1 features
	# convert text categories into integer features
	def get_word_features(self, sets, V, use_bigrams, use_trigrams):
		feats = numpy.zeros((len(V), len(sets)))
		labels = numpy.empty((1, len(sets)), dtype=int)

		for i, (p, c) in enumerate(sets):
			sys.stderr.write("Getting features for item "+str(i+1)+'\r')

#			tokens = word_tokenize(p.lower())
			tokens = self.mec.util.regex_tokenizer(p.lower())

			for w in tokens:
				if w in V: feats[V.index(w)][i] = 1

			if use_bigrams:
				bigrams = ngrams(tokens, 2)

				for b in bigrams:
					if b in V: feats[V.index(b)][i] = 1

			if use_trigrams:
				trigrams = ngrams(tokens, 3)

				for t in trigrams:
					for t in V: feats[V.index(t)][i] = 1

			if c == 'algebra': labels[0][i] = 0
			elif c == 'arithmetic': labels[0][i] = 1
			elif c == 'geometry': labels[0][i] = 2

		return feats, numpy.array(labels)

	def get_train_dependency_features(self, V):
		self.get_dependency_features(self.mec.train_problems)

	def get_test_dependency_features(self, V):
		self.get_dependency_features(self.mec.test_problems)

	def get_dependency_features(self, sets):
	##create hand features vector 
#		feats = numpy.zeros((len(hand_features), len(sets)))
		labels = numpy.empty((1, len(sets)), dtype=int)

		for i, (p, c) in sets:
			sys.stderr.write("Getting dependency features for item "+str(i+1)+'\r')

			doc = spacy.tokens.doc.Doc(nlp.vocab, words=self.mec.util.regex_tokenizer(p.decode('utf8'), True))

			for name, proc in nlp.pipeline:
				doc = proc(doc)

			tokens = self.mec.util.regex_tokenizer(p.lower())

			for word in doc:
				if re.match('\[M:', word.text):
					if word.i == 0:
						prev_tok_tag = 'null'
						prev_tok_dep = 'null'
						prev_tok_txt = 'null'
						next_tok_tag = word.nbor().tag_
						next_tok_dep = word.nbor().dep_
						next_tok_txt = word.nbor().lower_

					elif word.i == len(doc)-1:
						prev_tok_tag = word.nbor(-1).tag_
						prev_tok_dep = word.nbor(-1).dep_
						prev_tok_txt = word.nbor(-1).lower_
						next_tok_tag = 'null'
						next_tok_dep = 'null'
						next_tok_txt = 'null'

					else:
						prev_tok_tag = word.nbor(-1).tag_
						prev_tok_dep = word.nbor(-1).dep_
						prev_tok_txt = word.nbor(-1).lower_
						next_tok_tag = word.nbor().tag_
						next_tok_dep = word.nbor().dep_
						next_tok_txt = word.nbor().lower_

					sym_tri = (
						prev_tok_tag,#+'#'+prev_tok_dep, 
						word.tag_,#+'#'+word.dep_, 
						next_tok_tag#+'#'+next_tok_dep
					)

					spec_tri = (
						c,
						prev_tok_txt,
						word.text,
						next_tok_txt
					)

## if sym_tri or spec_tri in feats...yadda..
	#				if w in V: feats[hand_features.index(w)][i] = 1
			
			if c == 'algebra': labels[0][i] = 0
			elif c == 'arithmetic': labels[0][i] = 1
			elif c == 'geometry': labels[0][i] = 2

		return feats, numpy.array(labels)
