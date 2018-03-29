#!/usr/bin/python

import sys, re
import pickle, glob
import math, numpy
#from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from random import shuffle
from Utils import Utils as Util

class NaiveClassifier:
	alg = {}
	arith = {}
	geo = {}
	alg_count = 0.0
	alg_bigram_count = 0.0
	arith_count = 0.0
	arith_bigram_count = 0.0
	geo_count = 0.0
	geo_bigram_count = 0.0
	alg_problems = []
	arith_problems = []
	geo_problems = []
	categories = ['algebra', 'arithmetic', 'geometry']

	def __init__(self, pkl):
		self.avg_prec = 0.0
		self.avg_recall = 0.0

		self.util = Util()

		self.read_files(pkl)

	def read_files(self, save_pkl):
		if save_pkl:
			sys.stdout.write("Reading files for processing.\n")

			file_list = glob.glob('./processed/*.txt')

			sys.stdout.write("\tReading "+str(len(file_list))+" files.\n")

			for f_name in file_list:
				f = open(f_name, 'r').readlines()

				for l in f:
					#pid, 
					c, p = l.strip().split('#@#')

					if c == 'algebra':
						self.alg_count += 1
						self.alg_problems.append((p, c))
	
					elif c == 'arithmetic':
						self.arith_count += 1
						self.arith_problems.append((p, c))

					elif c == 'geometry':
						self.geo_count += 1
						self.geo_problems.append((p, c))

		else:
			prefix = './pickles/'

			self.alg_problems = pickle.load(open(prefix+'alg_problems.pkl', 'rb'))
			self.arith_problems = pickle.load(open(prefix+'arith_problems.pkl', 'rb'))
			self.geo_problems = pickle.load(open(prefix+'geo_problems.pkl', 'rb'))

			self.alg_count = len(self.alg_problems)
			self.arith_count = len(self.arith_problems)
			self.geo_count = len(self.geo_problems)

	def split_sets(self, pct, save_pkl):
		if save_pkl:
			shuffle(self.alg_problems)
			shuffle(self.arith_problems)
			shuffle(self.geo_problems)

		self.alg_train_count = int(self.alg_count * pct)
		self.alg_test_count = int(self.alg_count - self.alg_train_count)
		self.arith_train_count = int(self.arith_count * pct)
		self.arith_test_count = int(self.arith_count - self.arith_train_count)
		self.geo_train_count = int(self.geo_count * pct)
		self.geo_test_count = int(self.geo_count - self.geo_train_count)

		self.train_problems = self.alg_problems[self.alg_test_count:] + self.arith_problems[self.arith_test_count:] + self.geo_problems[self.geo_test_count:]
		self.test_problems = self.alg_problems[:self.alg_test_count] + self.arith_problems[:self.arith_test_count] + self.geo_problems[:self.geo_test_count]

		self.alg_train_set = self.alg_problems[self.alg_test_count:]
		self.alg_test_set = self.alg_problems[:self.alg_test_count]
		self.arith_train_set = self.arith_problems[self.arith_test_count:]
		self.arith_test_set = self.arith_problems[:self.alg_test_count]
		self.geo_train_set = self.geo_problems[self.geo_test_count:]
		self.geo_test_set = self.geo_problems[:self.geo_test_count]

	def compute_base_probs(self, k, use_bigrams, use_trigrams):
		self.alg, alg_bigram_count, alg_trigram_count = self.generate_counts(self.alg_train_set, use_bigrams, use_trigrams)
		self.arith, arith_bigram_count, arith_trigram_count = self.generate_counts(self.arith_train_set, use_bigrams, use_trigrams)
		self.geo, geo_bigram_count, geo_trigram_count = self.generate_counts(self.geo_train_set, use_bigrams, use_trigrams)

		self.p_alg = float(self.alg_train_count)/(self.alg_train_count+self.arith_train_count+self.geo_train_count)
		self.p_arith = float(self.arith_train_count)/(self.alg_train_count+self.arith_train_count+self.geo_train_count)
		self.p_geo = float(self.geo_train_count)/(self.alg_train_count+self.arith_train_count+self.geo_train_count)

		if use_bigrams:
			self.p_alg_bigram = float(alg_bigram_count)/(alg_bigram_count+arith_bigram_count+geo_bigram_count)
			self.p_arith_bigram = float(arith_bigram_count)/(alg_bigram_count+arith_bigram_count+geo_bigram_count)
			self.p_geo_bigram = float(geo_bigram_count)/(alg_bigram_count+arith_bigram_count+geo_bigram_count)

		if use_trigrams:
			self.p_alg_trigram = float(alg_trigram_count)/(alg_trigram_count+arith_trigram_count+geo_trigram_count)
			self.p_arith_trigram = float(arith_trigram_count)/(alg_trigram_count+arith_trigram_count+geo_trigram_count)
			self.p_geo_trigram = float(geo_trigram_count)/(alg_trigram_count+arith_trigram_count+geo_trigram_count)

		total_alg = float(sum(self.alg.values()))
		total_arith = float(sum(self.arith.values()))
		total_geo = float(sum(self.geo.values()))

		for w in self.alg:
			self.alg[w] /= total_alg

		for w in self.arith:
			self.arith[w] /= total_arith

		for w in self.geo:
			self.geo[w] /= total_geo

	# count all tokens in a given training set
	def generate_counts(self, t_set, b, t):
		counts = {}
		b_count = 0
		t_count = 0

		for p, c in t_set:
#			tokens = word_tokenize(p.lower())
			tokens = self.util.regex_tokenizer(p.lower())

			for w in tokens:
				if w not in counts: counts[w] = 1

				counts[w] += 1

			if b:
				bigrams = ngrams(tokens, 2)

				for b in bigrams:
					b_count += 1

					if b not in counts: counts[b] = 1

					counts[b] += 1

			if t:
				trigrams = ngrams(tokens, 3)

				for t in trigrams:
					t_count += 1

					if t not in counts: counts[t] = 1

					counts[t] += 1

		counts['<UNK>'] = 1.0
		if b: counts[('<UNK>', '<UNK>')] = 1.0
		if t: counts[('<UNK>', '<UNK>', '<UNK>')] = 1.0

		return counts, b_count, t_count

	def set_stats(self):
		print "Set Statistics:\n\tAlgebra Total/Train/Test: %d/%d/%d\n\tArithmetic Total/Train/Test: %d/%d/%d\n\tGeometry Total/Train/Test: %d/%d/%d" % (
			self.alg_count,
			self.alg_train_count,
			self.alg_test_count,
			self.arith_count,
			self.arith_train_count,
			self.arith_test_count,
			self.geo_count,
			self.geo_train_count,
			self.geo_test_count
		)

	def calculate_probs(self, use_bigrams, use_trigrams):
		confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

		for p, c in self.test_problems:
			alg_prob = math.log(self.p_alg+self.p_alg_bigram)
			arith_prob = math.log(self.p_arith+self.p_arith_bigram)
			geo_prob = math.log(self.p_geo+self.p_geo_bigram)

#			tokens = word_tokenize(p.lower())
			tokens = self.util.regex_tokenizer(p.lower())

			for w in tokens:
				if w in self.alg: alg_prob += math.log(self.alg[w])
				else: alg_prob += math.log(self.alg['<UNK>'])

				if w in self.arith: arith_prob += math.log(self.arith[w])
				else: arith_prob += math.log(self.arith['<UNK>'])

				if w in self.geo: geo_prob += math.log(self.geo[w])
				else: geo_prob += math.log(self.geo['<UNK>'])

			if use_bigrams:
				bigrams = ngrams(tokens, 2)

				for b in bigrams:
					if b in self.alg: alg_prob += math.log(self.alg[b])
					else: alg_prob += math.log(self.alg[('<UNK>', '<UNK>')])

					if b in self.arith: arith_prob += math.log(self.arith[b])
					else: arith_prob += math.log(self.arith[('<UNK>', '<UNK>')])

					if b in self.geo: geo_prob += math.log(self.geo[b])
					else: geo_prob += math.log(self.geo[('<UNK>', '<UNK>')])

			if use_trigrams:
				trigrams = ngrams(tokens, 3)

				for t in trigrams:
					if t in self.alg: alg_prob += math.log(self.alg[t])
					else: alg_prob += math.log(self.alg[('<UNK>', '<UNK>', '<UNK>')])

					if t in self.arith: arith_prob += math.log(self.arith[t])
					else: arith_prob += math.log(self.arith[('<UNK>', '<UNK>', '<UNK>')])

					if t in self.geo: geo_prob += math.log(self.geo[t])
					else: geo_prob += math.log(self.geo[('<UNK>', '<UNK>', '<UNK>')])

			if c == 'algebra':
				if alg_prob > arith_prob:
					if alg_prob > geo_prob:
						confusion_matrix[0][0] += 1

					elif geo_prob > arith_prob:
						confusion_matrix[0][2] += 1

					else:
						confusion_matrix[0][1] += 1

				elif arith_prob > geo_prob:
					confusion_matrix[0][1] += 1

				else:
					confusion_matrix[0][2] += 1

			elif c == 'arithmetic':
				if arith_prob > geo_prob:
					if arith_prob > alg_prob:
						confusion_matrix[1][1] += 1

					elif alg_prob > geo_prob:
						confusion_matrix[1][0] += 1

					else:
						confusion_matrix[1][2] += 1

				elif geo_prob > alg_prob:
					confusion_matrix[1][2] += 1

				else:
					confusion_matrix[1][0] += 1

			elif c == 'geometry':
				if geo_prob > alg_prob:
					if geo_prob > arith_prob:
						confusion_matrix[2][2] += 1

					elif arith_prob > alg_prob:
						confusion_matrix[2][1] += 1

					else:
						confusion_matrix[2][0] += 1

				elif alg_prob > arith_prob:
					confusion_matrix[2][0] += 1

				else:
					confusion_matrix[2][1] += 1

		return self.precision_recall(confusion_matrix)

	def precision_recall(self, cm):
		tp = cm[0][0] + cm[1][1] + cm[2][2]
		fp = cm[0][1] + cm[2][1]
		fn = cm[1][0] + cm[1][2]

		print "TP: %d, FP: %d, FN: %d" % (tp, fp, fn)

		return list([float(tp)/(tp+fp), float(tp)/(tp+fn)])
