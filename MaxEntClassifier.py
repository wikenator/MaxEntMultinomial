#!/usr/bin/python

'''
Author: Arnold Wikey
Date: 2018
Description: class containing maximum entropy (logisitic regression) functions. This version of MaxEnt implements the softmax activation function, cross entropy cost function, and gradient descent with L2 regularization.
'''

import sys, math
import warnings
import numpy
from nltk.util import ngrams
from NaiveClassifier import NaiveClassifier as NBC

class MaxEntClassifier(NBC):
	# MaxEntClassifier extends NaiveClassifier
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

	# gradient descent with L2 regularization
	def gradient_reg(self, f, l, p, w, reg_lambda):
		return (
			numpy.dot(
				f, 
				(p - self.util.onehot_enc(l))
			.T) - \
			reg_lambda * \
			numpy.linalg.norm(w)
		).T

	# training for maxent classification during grid search
	# returns optimized weights, minimum cost value, and final learning rate
	def maxent_grid_search(self, f, v, w, l, vl, n_steps=1000, learn_rate=5e-4, reg_coeff=0.001, threshold=1e-6):
		warnings.filterwarnings('error')
		numpy.seterr(all='raise')

		gstop = 0
		astop = 0
		nostop = False
		accuracies = []
		costs = []
		v_costs = []

		sys.stdout.write("\nRunning MaxEnt classification.\n")

		probabilities = numpy.dot(w, f)

		try:
			predictions = self.softmax(probabilities)

		except FloatingPointError:
			sys.stdout.write('\nBad numbers used.\n')
			return 'no', 'no', 'no'

		c = self.cost(predictions, l)

		validate_prob = numpy.dot(w, v)

		try:
			validate_pred = self.softmax(validate_prob)

		except FloatingPointError:
			sys.stdout.write('\nBad numbers used.\n')
			return 'no', 'no', 'no'

		vc = self.cost(validate_pred, vl)
		
		class_bin = self.hard_classify(predictions)
		accuracy = ((class_bin == l).sum().astype(float)/len(class_bin))
		accuracies.append(accuracy)

		for i in xrange(n_steps):
			costs.append(c)
			v_costs.append(vc)
			gc = vc / min(v_costs)

			sys.stderr.write("iter: %d cost: %.9f acc: %.9f gen: %.9f\r" % (i+1, c, accuracy, gc))

			grad = self.gradient_reg(f, l, probabilities, w, reg_coeff)
			w -= learn_rate * grad

			probabilities = numpy.dot(w, f)

			try:
				predictions = self.softmax(probabilities)

			except FloatingPointError:
				sys.stdout.write('\nBad numbers used.\n')
				return 'no', 'no', 'no'

			new_cost = self.cost(predictions, l)

			old_acc = float(sum(accuracies))/len(accuracies)
			class_bin = self.hard_classify(predictions)
			accuracy = ((class_bin == l).sum().astype(float)/len(class_bin))
			accuracies.append(accuracy)
			new_acc = float(sum(accuracies))/len(accuracies)
			acc_diff = abs(new_acc-old_acc)

			if math.isnan(new_cost):
				return 'div', 'div', 'div'

			if nostop == False:
				if acc_diff >= 0 and acc_diff < 0.001:
					if astop == 5:
						sys.stdout.write("\nAccuracy improvement no longer significant.")
						break

					else:
						astop += 1
			
				else:
					astop = 0

				if gc > 1.6:
					if gstop == 5:
						sys.stdout.write("\nGeneral cost improvement no longer significant.")
						break

					else:
						gstop += 1

				else:
					gstop = 0

				# stop gradient descent if new cost is not much better
				if abs(c-new_cost) < threshold:
					break

				# lower learning rate if diverging
				elif c-new_cost < 0:
					w += learn_rate * grad
					learn_rate *= 0.5

				# increase learning rate if converging
				elif c-new_cost > 0:
					learn_rate *= 1.05

			validate_prob = numpy.dot(w, v)

			try:
				validate_pred = self.softmax(validate_prob)

			except FloatingPointError:
				sys.stdout.write('\nBad numbers used.\n')
				return 'no', 'no', 'no'

			vc = self.cost(validate_pred, vl)
			c = new_cost

		sys.stdout.write('\n')
		print

		return w, c, learn_rate
		
	# training for maxent classification
	# returns optimized weights, minimum cost value, and final learning rate
	def maxent(self, f, w, l, n_steps=1000, learn_rate=5e-4, reg_coeff=0.001, threshold=1e-6):
		warnings.filterwarnings('error')
		numpy.seterr(all='raise')

		astop = 0
		accuracies = []
		costs = []

		f_train = open('terr.csv', 'w')
		f_val = open('verr.csv', 'w')
		f_acc = open('acc.csv', 'w')

		sys.stdout.write("\nRunning MaxEnt classification.\n")

		probabilities = numpy.dot(w, f)
		predictions = self.softmax(probabilities)
		c = self.cost(predictions, l)

		class_bin = self.hard_classify(predictions)
		accuracy = ((class_bin == l).sum().astype(float)/len(class_bin))
		accuracies.append(accuracy)

		for i in xrange(n_steps):
			f_train.write('%d,%.9f\n' % (i, c))
			f_val.write('%d,%.9f\n' % (i, vc))
			f_acc.write('%d,%.9f\n' % (i, accuracy))

			costs.append(c)

			grad = self.gradient_reg(f, l, probabilities, w, reg_coeff)
			w -= learn_rate * grad
			probabilities = numpy.dot(w, f)
			predictions = self.softmax(probabilities)
			new_cost = self.cost(predictions, l)

			old_acc = float(sum(accuracies))/len(accuracies)
			class_bin = self.hard_classify(predictions)
			accuracy = ((class_bin == l).sum().astype(float)/len(class_bin))
			accuracies.append(accuracy)
			new_acc = float(sum(accuracies))/len(accuracies)
			acc_diff = abs(new_acc-old_acc)

			if math.isnan(new_cost):
				return 'div', 'div', 'div'

			if acc_diff >= 0 and acc_diff < 0.001:
				if astop == 5:
					sys.stdout.write("\nAccuracy improvement no longer significant.")
					break

				else:
					astop += 1
			
			else:
				astop = 0

			# stop gradient descent if new cost is not much better
			if abs(c-new_cost) < threshold:
				break

			# lower learning rate if diverging
			elif c-new_cost < 0:
				w += learn_rate * grad
				learn_rate *= 0.5

			# increase learning rate if converging
			elif c-new_cost > 0:
				learn_rate *= 1.05

			c = new_cost

		sys.stdout.write('\n')

		f_train.close()
		f_val.close()
		f_acc.close()

		print

		return w, c, learn_rate
		
	# used for future precision and recall calculations
	def maxent_confusion_matrix(self, actual, predicted):
		confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

		for i, a in enumerate(actual):
			confusion_matrix[a][predicted[i]] += 1

		return confusion_matrix
	# used for future precision and recall calculations
	def maxent_confusion_matrix(self, actual, predicted):
		confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

		for i, a in enumerate(actual):
			confusion_matrix[a][predicted[i]] += 1

		return confusion_matrix
