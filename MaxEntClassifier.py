#!/usr/bin/python

import sys, math
import numpy
#from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from NaiveClassifier import NaiveClassifier as NBC

class MaxEntClassifier(NBC):
	def __init__(self, pkl):
		NBC.__init__(self, pkl)

		self.train_acc = []
		self.test_acc = []
		self.precision = []
		self.recall = []

	# convert probability vector into 0-1 hard classification
	def hard_classify(self, vec):
		return vec.argmax(axis=0)

	# standard softmax function
	def softmax(self, z):
		return (numpy.exp(z) / numpy.sum(numpy.exp(z), axis=0))

	# calculates cross entropy between predicted and actual set labels
	def cross_entropy(self, predicted, actual):
		actual = self.util.onehot_enc(actual)

		return -numpy.sum(numpy.log(predicted) * actual, axis=0)

	# cost minimization function of cross entropy
	def cost(self, predicted, actual):
		return numpy.mean(self.cross_entropy(predicted, actual))

	# gradient descent
	def gradient(self, f, l, p):
		return numpy.dot(f, (p - self.util.onehot_enc(l)).T).T

	# gradient descent
	def gradient_reg(self, f, l, p, w, reg_lambda):
#		return numpy.mean(numpy.dot(f, (p - self.util.onehot_enc(l)).T))
		return (
			numpy.dot(
				f, 
				(p - self.util.onehot_enc(l))
			.T) - \
			reg_lambda * \
			numpy.linalg.norm(w)
		).T

	# training for maxent classification
	# returns optimized weights, minimum cost value, and final learning rate
	def maxent(self, f, w, l, n_steps=1000, learn_rate=5e-4, reg_coeff=0.001, threshold=1e-5):
		sys.stderr.write("\nRunning MaxEnt classification.\n")

#		c = self.cost(self.softmax(numpy.dot(w, f)), l)

		probabilities = numpy.dot(w, f)
		predictions = self.softmax(probabilities)
		c = self.cost(predictions, l)
		
		for i in xrange(n_steps):
			sys.stderr.write("iter: "+str(i+1)+" cost: "+str(c)+"\r")

#			w -= learn_rate * self.gradient(f, l, probabilities)
			grad = self.gradient_reg(f, l, probabilities, w, reg_coeff)

			w -= learn_rate * grad

			probabilities = numpy.dot(w, f)
			predictions = self.softmax(probabilities)
			new_cost = self.cost(predictions, l)

			if math.isnan(new_cost):
				sys.exit("\nGradient descent has diverged. Last learn rate: "+str(learn_rate)+"\n")

			else:
				# stop gradient descent if new cost is not much better
				if abs(c-new_cost) < threshold:
					break

				# lower learning rate if diverging
				elif c-new_cost < 0:
					w += learn_rate * grad
					learn_rate /= 2

				# increase learning rate if converging
				elif c-new_cost > 0:
					learn_rate *= 1.05

				c = new_cost

		print

		return w, c, learn_rate

	def maxent_confusion_matrix(self, actual, predicted):
		confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

		for i, a in enumerate(actual):
			confusion_matrix[a][predicted[i]] += 1

		return confusion_matrix
