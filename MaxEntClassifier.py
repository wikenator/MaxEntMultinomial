#!/usr/bin/python

import sys, math
import numpy
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from NaiveClassifier import NaiveClassifier as NBC

class MaxEntClassifier(NBC):
	def __init__(self, pkl):
		NBC.__init__(self, pkl)

		# create array of selected stop words and punctuation
		self.stop_words = set(stopwords.words('english'))
		self.stop_words.add(',')
		self.stop_words.add('.')
		self.stop_words.add('?')
		self.stop_words.add(';')
		self.stop_words.add(':')
		self.stop_words.add(')')
		self.stop_words.add('(')
		self.stop_words.add('$')

	# add all word tokens into a set and sort tokens alphabetically
	def get_vocabulary(self):
		sys.stderr.write("Compiling vocabulary.\n")

		all_problems = self.alg_problems + self.arith_problems + self.geo_problems
		vocab = [w.lower() for p in all_problems for w in word_tokenize(p[0]) if (not w in self.stop_words) and (len(w) > 2)]
		vocab += [b for p in all_problems for b in ngrams(word_tokenize(p[0].lower()), 2)]
		vocab = set(vocab)
		vocab = list(vocab)
		vocab.sort()

		return vocab

	# set initial weights to raw word count probabilities
	def get_init_weights(self, V, use_bigrams=True):
		sys.stderr.write("Initializing weights.\n")

		wts = numpy.zeros((3, len(V)))
		
		# process algebra problem set
		for p, c in self.alg_train_set:
			tokens = word_tokenize(p.lower())

			for w in tokens:
				if w in V: wts[0][V.index(w)] = self.alg[w]

			if use_bigrams:
				bigrams = ngrams(tokens, 2)

				for b in bigrams:
					if b in V: wts[0][V.index(b)] = self.alg[b]

		# process arithmetic problem set
		for p, c in self.arith_train_set:
			tokens = word_tokenize(p.lower())

			for w in tokens:
				if w in V: wts[1][V.index(w)] = self.arith[w]

			if use_bigrams:
				bigrams = ngrams(tokens, 2)

				for b in bigrams:
					if b in V: wts[1][V.index(b)] = self.arith[b]

		# process geometry problem set
		for p, c in self.geo_train_set:
			tokens = word_tokenize(p.lower())

			for w in tokens:
				if w in V: wts[2][V.index(w)] = self.geo[w]

			if use_bigrams:
				bigram = ngrams(tokens, 2)

				for b in bigrams:
					if b in V: wts[2][V.index(b)] = self.geo[b]

		return wts	

	# abstract for get_features
	def get_train_features(self, V, use_bigrams=True):
		sys.stderr.write("\nVectorizing training features.\n")

		train_sets = self.alg_train_set + self.arith_train_set + self.geo_train_set

		return self.get_features(train_sets, V, use_bigrams)

	# abstract for get_features
	def get_test_features(self, V, use_bigrams=True):
		sys.stderr.write("\nVectorizing test features.\n")

		test_sets = self.alg_test_set + self.arith_test_set + self.geo_test_set

		return self.get_features(test_sets, V, use_bigrams)

	# convert word tokens into 0-1 features
	# convert text categories into integer features
	def get_features(self, sets, V, use_bigrams):
		feats = numpy.zeros((len(V), len(sets)))
		labels = numpy.empty((1, len(sets)), dtype=int)

		for i, (p, c) in enumerate(sets):
			tokens = word_tokenize(p.lower())

			for w in tokens:
				if w in V: feats[V.index(w)][i] = 1

			if use_bigrams:
				bigrams = ngrams(tokens, 2)

				for b in bigrams:
					if b in V: feats[V.index(b)][i] = 1

			if c == 'algebra': labels[0][i] = 0
			elif c == 'arithmetic': labels[0][i] = 1
			elif c == 'geometry': labels[0][i] = 2

		return feats, numpy.array(labels)

	# convert probability vector into 0-1 hard classification
	def hard_classify(self, vec):
		return vec.argmax(axis=0)

	# standard softmax function
	def softmax(self, z):
		return (numpy.exp(z) / numpy.sum(numpy.exp(z), axis=0))

	# calculates cross entropy between predicted and actual set labels
	def cross_entropy(self, predicted, actual):
		actual = self.onehot_enc(actual)

		return -numpy.sum(numpy.log(predicted) * actual, axis=0)

	# convert 1-row vector to one-hot vector
	def onehot_enc(self, vec):
		onehot = []

		for val in vec[0]:
			l = [0 for i in range(len(self.categories))]
			l[val] = 1
			onehot.append(l)
		
		return numpy.asarray(onehot).T

	# cost minimization function of cross entropy
	def cost(self, predicted, actual):
		return numpy.mean(self.cross_entropy(predicted, actual))

	# gradient descent
	def gradient(self, f, l, p):
		return numpy.dot(f, (p - self.onehot_enc(l)).T).T

	# gradient descent
	def gradient_reg(self, f, l, p, w, reg_lambda):
#		return numpy.mean(numpy.dot(f, (p - self.onehot_enc(l)).T))
		return (
			numpy.dot(
				f, 
				(p - self.onehot_enc(l))
			.T) - \
			reg_lambda * \
			numpy.linalg.norm(w)
		).T

	# training for maxent classification
	# returns optimized weights, minimum cost value, and final learning rate
	def maxent(self, f, w, l, n_steps=1000, learn_rate=5e-4, reg_coeff=0.001, threshold=1e-4):
		sys.stderr.write("\nRunning MaxEnt classification.\n")

#		c = self.cost(self.softmax(numpy.dot(w, f)), l)

		probabilities = numpy.dot(w, f)
		predictions = self.softmax(probabilities)
		c = self.cost(predictions, l)
		
		for i in xrange(n_steps):
			sys.stderr.write("iter: "+str(i)+" cost: "+str(c)+"\r")

#			w -= learn_rate * self.gradient(f, l, probabilities)
			w -= learn_rate * self.gradient_reg(f, l, probabilities, w, reg_coeff)

			probabilities = numpy.dot(w, f)
			predictions = self.softmax(probabilities)
			new_cost = self.cost(predictions, l)

			if math.isnan(new_cost):
				sys.exit("\nGradient descent has diverged. Last learn rate: "+str(learn_rate)+"\n")

			else:
				# stop gradient descent if new cost is not much better
				if abs(c-new_cost) < threshold: break
				# lower learning rate if gradient descent not converging
				elif abs(c-new_cost) > 0.5: learn_rate /= 2

				c = new_cost

		print

		return w, c, learn_rate
