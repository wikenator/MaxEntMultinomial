#!/usr/bin/python

import os, sys, pickle, argparse
import numpy
from MaxEntClassifier import MaxEntClassifier as MEC

def NB_Test(m, b, t):
	for i in range(0, 5):
		m.split_sets(0.8)
		prec, recall = m.calculate_probs(b, t)

		print "Pass %d\n\tPrecision: %.4f\n\tRecall: %.4f" % (i+1, prec, recall)
		m.avg_prec += prec
		m.avg_recall += recall

	print "Precision: %.4f\nRecall: %.4f" % (
		m.avg_prec/mec.iters, 
		m.avg_recall/mec.iters
	)

def pickle_objs(m, prefix, v, w, f, l):
	sys.stderr.write('\nPickling objects.\n')

	pkl_vocab = open(prefix+'all_words.pkl', 'wb')
	pickle.dump(v, pkl_vocab)
	pkl_vocab.close()
	
	pkl_wts = open(prefix+'init_weights.pkl', 'wb')
	pickle.dump(w, pkl_wts)
	pkl_wts.close()

	pkl_alg = open(prefix+'alg_problems.pkl', 'wb')
	pickle.dump(m.alg_problems, pkl_alg)
	pkl_alg.close()

	pkl_arith = open(prefix+'arith_problems.pkl', 'wb')
	pickle.dump(m.arith_problems, pkl_arith)
	pkl_arith.close()

	pkl_geo = open(prefix+'geo_problems.pkl', 'wb')
	pickle.dump(m.geo_problems, pkl_geo)
	pkl_geo.close()

	pkl_train_feat = open(prefix+'train_features.pkl', 'wb')
	pickle.dump(f, pkl_train_feat)
	pkl_train_feat.close()

	pkl_train_label = open(prefix+'train_labels.pkl', 'wb')
	pickle.dump(l, pkl_train_label)
	pkl_train_label.close()

if __name__ == '__main__':
	cmd_line_parser = argparse.ArgumentParser(description='Train MaxEnt classifier to classify math word problems.')
	cmd_line_parser.add_argument('-b', '--use_bigrams', action='store_true', help='Add bigram features to maxent learning.')
	cmd_line_parser.add_argument('-t', '--use_trigrams', action='store_true', help='Add trigram features to maxent learning.')
#	pkl_group = cmd_line_parser.add_mutually_exclusive_group(required=False)
	cmd_line_parser.add_argument('-l', '--load_pickle', action='store_true', help='Load data from pickle files (will not recalculate data).')
	cmd_line_parser.add_argument('-r', '--no_retrain', action='store_true', help='Do not retrain weights from loaded data.')
#	pkl_group.add_argument('-s', '--save_pickle', action='store_true', help='Save data to pickle files after calculation.')
	cmd_line_parser.add_argument('--steps', nargs=1, default=1000, help='Number of iterations for maxent gradient descent calculated during learning. Default: %(default)s')
	cmd_line_parser.add_argument('--learn_rate', nargs=1, default=5e-4, help='Learning rate to use during maxent learning. Default: %(default)s')
	cmd_line_parser.add_argument('--reg_coeff', nargs=1, default=0.001, help='Regularization coefficient to normalize maxent gradient descent during learning. Default: %(default)s')
	args = cmd_line_parser.parse_args()
	
	sys.stderr.write("MaxEnt parameters:\n")
	print('\t' + str(args) + '\n')

	save_pickle = not args.load_pickle

	if type(args.steps) == list:
		n_steps = int(args.steps[0])

	else:
		n_steps = int(args.steps)

	if type(args.learn_rate) == list:
		learn_rate = float(args.learn_rate[0])

	else:
		learn_rate = float(args.learn_rate)

	if type(args.reg_coeff) == list:
		reg_coeff = float(args.reg_coeff[0])

	else:
		reg_coeff = float(args.reg_coeff)

	prefix = './pickles/'

	mec = MEC(save_pickle)
	mec.iters = 5
	k = 100

	if len(sys.argv) > 1 and sys.argv[1] == 'nb':
		NB_Test(mec, args.use_bigrams, args.use_trigrams)

	mec.split_sets(0.8, save_pickle)
	mec.compute_base_probs(k, args.use_bigrams, args.use_trigrams)
	mec.set_stats()

	if save_pickle:
		all_words = mec.get_vocabulary(args.use_bigrams, args.use_trigrams)
		weights = mec.get_init_weights(all_words, args.use_bigrams, args.use_trigrams)
		train_features, train_labels = mec.get_train_features(all_words, args.use_bigrams, args.use_trigrams)

#		if not os.path.exists('./pickles/config'+str(args.use_bigrams)+str(args.use_trigrams)+'.cfg'):
#			pkl_files = os.listdir('./pickles/')

#			for f in pkl_files:
#				if f.endswith('.cfg'):
#					os.remove(os.path.join('./pickles/', f))

#			open('./pickles/config'+str(args.use_bigrams)+str(args.use_trigrams)+'.cfg', 'w').close()

		pickle_objs(
			mec, 
			prefix, 
			all_words, 
			weights, 
			train_features, 
			train_labels
		)
	
	else:
		sys.stderr.write("Reading pickle files.\n")

		all_words = pickle.load(open(prefix+'all_words.pkl', 'rb'))
		weights = pickle.load(open(prefix+'init_weights.pkl', 'rb'))
		train_features = pickle.load(open(prefix+'train_features.pkl', 'rb'))
		train_labels = pickle.load(open(prefix+'train_labels.pkl', 'rb'))

	# run maxent classifier
	# default values: n_steps=1000
	#		  learn_rate=5e-4
	#		  reg_coeff=0.001
	#		  threshold=1e-5
	if args.no_retrain:
		weights = pickle.load(open('./data/train_weights.pkl', 'rb'))

	else:
		weights, min_cost, best_learn_rate = mec.maxent(
			train_features, 
			weights, 
			train_labels, 
			n_steps,
			learn_rate,
			reg_coeff
		)

		wts = open('./data/train_weights.pkl', 'wb')
		pickle.dump(weights, wts)
		wts.close()
		
	class_prob_train = numpy.dot(weights, train_features)
	class_bin_train = mec.hard_classify(class_prob_train)

	print "\nTraining Accuracy: %.4f" % \
		((class_bin_train == train_labels).sum().astype(float)/len(class_bin_train))

	test_features, test_labels = mec.get_test_features(all_words, args.use_bigrams, args.use_trigrams)
	class_prob_test = numpy.dot(weights, test_features)
	class_bin_test = mec.hard_classify(class_prob_test)

	print "\nTest Accuracy: %.4f" % \
		((class_bin_test == test_labels) \
		.sum() \
		.astype(float) / \
		len(class_bin_test))

	if not args.no_retrain:
		print "\nMaxEnt Stats:\n\tMinimized Cost: %.6f\n\tBest Learning Rate: %.6f" % (min_cost, best_learn_rate)

	print "\nTop 10 important features:\n\t"

	for k, v in sorted(mec.alg.iteritems(), key=lambda(k, v): (v, k), reverse=True)[:10]:
		print "%s: %f" % (k, v)
