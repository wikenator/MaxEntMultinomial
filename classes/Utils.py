#!/usr/bin/python

'''
Author: Arnold Wikey
Date: 2018
Description: class containing functions used across all other classes (NaiveClassifier, MaxEntClassifier, and FeatureBuilder).
'''

import sys, re
import pickle, argparse
import numpy
from nltk import RegexpTokenizer as RT
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class Utils():
	def __init__(self):
		self.categories = ('algebra', 'arithmetic', 'geometry')

		# regex pattern used for handling abbreviations
		self.abbrev_pattern = re.compile(r'^(in|cm|ft|d|s|bu|mi|km|m|oz|yd)$')

		# create array of selected stop words and punctuation
		self.stop_words = set(stopwords.words('english'))
		self.stop_words.add(',')
		self.stop_words.add('.')
		self.stop_words.add('?')
		self.stop_words.add(';')
		self.stop_words.add(':')
		self.stop_words.add('$')

	# parse command line arguments
	def cmdline_argparse(self):
		cmd_line_parser = argparse.ArgumentParser(description='Train MaxEnt classifier to classify math word problems.')
		cmd_line_parser.add_argument('-u', '--use_unigrams', action='store_true', help='Add unigram features to maxent learning.')
		cmd_line_parser.add_argument('-b', '--use_bigrams', action='store_true', help='Add bigram features to maxent learning.')
		cmd_line_parser.add_argument('-t', '--use_trigrams', action='store_true', help='Add trigram features to maxent learning.')
		cmd_line_parser.add_argument('-d', '--dep_parse', action='store_true', help='Add dependency parsing features to maxent learning.')
		#pkl_group = cmd_line_parser.add_mutually_exclusive_group(required=False)
		cmd_line_parser.add_argument('-p', '--save_pickle', action='store_true', help='Save data to pickle files.')
		cmd_line_parser.add_argument('-l', '--load_pickle', action='store_true', help='Load data from pickle files (will not recalculate data).')
		cmd_line_parser.add_argument('-n', '--naive', action='store_true', help='Run Naive Bayes classifer before MaxEnt.')
		cmd_line_parser.add_argument('-R', '--no_retrain', action='store_true', help='Do not retrain weights from loaded data.')
		#pkl_group.add_argument('-s', '--save_pickle', action='store_true', help='Save data to pickle files after calculation.')
		cmd_line_parser.add_argument('-s', '--steps', nargs=1, default=1000, help='Number of iterations for maxent gradient descent calculated during learning. Default: %(default)s')
		cmd_line_parser.add_argument('-r', '--learn_rate', nargs=1, default=7e-5, help='Learning rate to use during maxent learning. Default: %(default)s')
		cmd_line_parser.add_argument('-c', '--reg_coeff', nargs=1, default=0.5, help='Regularization coefficient to normalize maxent gradient descent during learning. Default: %(default)s')
		cmd_line_parser.add_argument('-g', '--grid_search', action='store_true', help='Searches 2,025 learning rate and regulation coefficient combinations to find the best values during maxent learning. Uses the best trained values for the provided test set.')
		cmd_line_parser.add_argument('-f', '--folds', nargs=1, default=1, help='Perform k-fold cross-validation. Larger k = less bias, more variance. Smaller k = more bias, less variance. Accuracy from each cross-validation will be averaged over all folds. Default: %(default)s')
		cmd_line_parser.add_argument('--seed', nargs=1, default=1, help='Set seed for random data shuffle. Use same seed when saving and loading pickle files. Default: %(default)s')
		args = cmd_line_parser.parse_args()

		if type(args.steps) == list: args.steps = int(args.steps[0])
		else: args.steps = int(args.steps)

		if type(args.learn_rate) == list: args.learn_rate = float(args.learn_rate[0])
		else: args.learn_rate = float(args.learn_rate)

		if type(args.reg_coeff) == list: args.reg_coeff = float(args.reg_coeff[0])
		else: args.reg_coeff = float(args.reg_coeff)

		if type(args.folds) == list: args.folds = int(args.folds[0])
		else: args.folds = int(args.folds)

		if type(args.seed) == list: args.seed = int(args.seed[0])
		else: args.seed = int(args.seed)

		if not(args.use_unigrams or args.use_bigrams or args.use_trigrams or args.dep_parse): args.use_unigrams = True

		sys.stdout.write("MaxEnt parameters:\n")
		print('\t' + str(args) + '\n')

		return args

	# save data as pickle object
	def pickle_objs(self, prefix, i, data):
		sys.stdout.write('\nPickling objects.\n')

		for k, v in data.iteritems():
			pkl_fh = open(prefix + k + str(i) + '.pkl', 'wb')
			pickle.dump(v, pkl_fh)
			pkl_fh.close()

	# tokenize sentences based on regular expression
	# nltk word_tokenize does not tokenize math tokens correctly
	def regex_tokenizer(self, sent, whole_sent=False):
		regex_tokenizer = RT('\w+|\[M:.*?\]|[\(\)\.\,;\?\!]|\S+')
		tokens = regex_tokenizer.tokenize(sent)
		i = 0
		j = len(tokens)-1

		# combine abbreviations with their period;
		# separation is a result of tokenizing the sentence
		while i < j:
			if re.match(self.abbrev_pattern, tokens[i]) and tokens[i+1] == '.':
				tokens[i:i+2] = [''.join(tokens[i:i+2])]
				j -= 1

			i += 1

		# return tokenized sentence minus stopword and short words
		if not whole_sent:
			return [t for t in tokens if not t in self.stop_words and len(t) > 2]

		# return entire tokenized sentence
		else:
			return [t for t in tokens]

	# convert 1-row vector to one-hot vector
	def onehot_enc(self, vec):
		onehot = []

		for val in vec[0]:
			l = [0 for i in range(len(self.categories))]
			l[val] = 1
			onehot.append(l)

		return numpy.asarray(onehot).T
