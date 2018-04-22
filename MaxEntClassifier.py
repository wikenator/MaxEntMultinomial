#!/usr/bin/python

'''
Author: Arnold Wikey
Date: 2018
Description: class containing maximum entropy (logisitic regression) functions. This version of MaxEnt implements the softmax activation function, cross entropy cost function, and gradient descent with L2 regularization.
'''

import sys, math
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

	# training for maxent classification
	# returns optimized weights, minimum cost value, and final learning rate
#	def maxent(self, f, w, l, n_steps=1000, learn_rate=5e-4, reg_coeff=0.001, threshold=1e-5):
	def maxent(self, f, v, w, l, vl, n_steps=1000, learn_rate=5e-4, reg_coeff=0.001, threshold=1e-6):
		gstop = 0
		astop = 0
		nostop = False
		accuracies = []
		costs = []
		v_costs = []

		f_train = open('terr.csv', 'w')
		f_val = open('verr.csv', 'w')
		f_gen = open('gerr.csv', 'w')
		f_acc = open('acc.csv', 'w')

		sys.stderr.write("\nRunning MaxEnt classification.\n")

		probabilities = numpy.dot(w, f)
		predictions = self.softmax(probabilities)
		c = self.cost(predictions, l)

		validate_prob = numpy.dot(w, v)
		validate_pred = self.softmax(validate_prob)
		vc = self.cost(validate_pred, vl)
		v_costs.append(vc)
		gc = vc / min(v_costs)
		
		class_bin = self.hard_classify(predictions)
		accuracy = ((class_bin == l).sum().astype(float)/len(class_bin))
		accuracies.append(accuracy)

		for i in xrange(n_steps):
			f_train.write('%d,%.9f\n' % (i, c))
			f_val.write('%d,%.9f\n' % (i, vc))
			f_gen.write('%d,%.9f\n' % (i, gc))
			f_acc.write('%d,%.9f\n' % (i, accuracy))

			costs.append(c)

			sys.stderr.write("iter: %d cost: %.9f acc: %.9f gen: %.9f\r" % (i+1, c, accuracy, gc))

			# check stopping criteria every 10 epochs
#			if (i+1) % 10 == 0:
#				min_cost = min(costs[i-10+1:])
#				progress = 1000 * (sum(costs[i-10+1:]) / (10 * min_cost) - 1)

#				sys.stderr.write("iter: "+str(i+1)+" cost: "+str(c)+" acc: "+str(accuracy)+"\n\tgen: "+str(gen_cost)+"\n\tprog: "+str(progress)+"\n\tsc: "+str(gen_cost/progress)+"\n")

#				if gen_cost / progress > 1.5:
#					sys.stdout.write("\nProgress slowing down.")
#					break

			grad = self.gradient_reg(f, l, probabilities, w, reg_coeff)
			w -= learn_rate * grad

			probabilities = numpy.dot(w, f)
			predictions = self.softmax(probabilities)
			new_cost = self.cost(predictions, l)

			validate_prob = numpy.dot(w, v)
			validate_pred = self.softmax(validate_prob)
			new_vc = self.cost(validate_pred, vl)
			v_costs.append(new_vc)
			new_gc = new_vc / min(v_costs)

			old_acc = float(sum(accuracies))/len(accuracies)
			class_bin = self.hard_classify(predictions)
			accuracy = ((class_bin == l).sum().astype(float)/len(class_bin))
			accuracies.append(accuracy)
#			acc_diff = float(sum(accuracies))/len(accuracies)-accuracy
			new_acc = float(sum(accuracies))/len(accuracies)
			acc_diff = new_acc-old_acc

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

#				if gc > 1 and new_gc > 1 and abs(gc-new_gc) < 0.01:
#					if gstop == 5:
#						sys.stdout.write("\nGeneral cost improvement no longer significant.")
#						break

#					else:
#						gstop += 1

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
			gc = new_gc
			vc = new_vc

		sys.stdout.write('\n')

		f_train.close()
		f_val.close()
		f_gen.close()

		print

		return w, c, learn_rate
		
	# used for future precision and recall calculations
	def maxent_confusion_matrix(self, actual, predicted):
		confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

		for i, a in enumerate(actual):
			confusion_matrix[a][predicted[i]] += 1

		return confusion_matrix
