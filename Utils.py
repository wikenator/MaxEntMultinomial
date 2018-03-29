#!/usr/bin/python

import sys
import pickle, argparse
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

	def cmdline_argparse(self):
		cmd_line_parser = argparse.ArgumentParser(description='Train MaxEnt classifier to classify math word problems.')
		cmd_line_parser.add_argument('-b', '--use_bigrams', action='store_true', help='Add bigram features to maxent learning.')
		cmd_line_parser.add_argument('-t', '--use_trigrams', action='store_true', help='Add trigram features to maxent learning.')
		#pkl_group = cmd_line_parser.add_mutually_exclusive_group(required=False)
		cmd_line_parser.add_argument('-l', '--load_pickle', action='store_true', help='Load data from pickle files (will not recalculate data).')
		cmd_line_parser.add_argument('-r', '--no_retrain', action='store_true', help='Do not retrain weights from loaded data.')
		#pkl_group.add_argument('-s', '--save_pickle', action='store_true', help='Save data to pickle files after calculation.')
		cmd_line_parser.add_argument('--steps', nargs=1, default=1000, help='Number of iterations for maxent gradient descent calculated during learning. Default: %(default)s')
		cmd_line_parser.add_argument('--learn_rate', nargs=1, default=5e-4, help='Learning rate to use during maxent learning. Default: %(default)s')
		cmd_line_parser.add_argument('--reg_coeff', nargs=1, default=0.001, help='Regularization coefficient to normalize maxent gradient descent during learning. Default: %(default)s')
		args = cmd_line_parser.parse_args()

		if type(args.steps) == list:
			args.steps = int(args.steps[0])

		else:
			args.steps = int(args.steps)

		if type(args.learn_rate) == list:
			args.learn_rate = float(args.learn_rate[0])

		else:
			args.learn_rate = float(args.learn_rate)

		if type(args.reg_coeff) == list:
			args.reg_coeff = float(args.reg_coeff[0])

		else:
			args.reg_coeff = float(args.reg_coeff)

		sys.stderr.write("MaxEnt parameters:\n")
		print('\t' + str(args) + '\n')

		return args

	def pickle_objs(self, prefix, data):
		sys.stderr.write('\nPickling objects.\n')

		for k, v in data.iteritems():
			pkl_fh = open(prefix + k + '.pkl', 'wb')
			pickle.dump(v, pkl_fh)
			pkl_fh.close()

	def regex_tokenizer(self, sent):
		regex_tokenizer = RT('\w+|\[M:.*?\]|\S+')
		tokens = regex_tokenizer.tokenize(sent)

		return [t for t in tokens if not t in self.stop_words and len(t) > 2]
