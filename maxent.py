#!/usr/bin/python

import os, sys, pickle#, argparse
import numpy
from MaxEntClassifier import MaxEntClassifier as MEC
from Utils import Utils as Util

def NB_Test(m, b, t):
	for i in xrange(mec.iters):
		m.split_sets(0.8)
		prec, recall = m.calculate_probs(b, t)

		print "Pass %d\n\tPrecision: %.4f\n\tRecall: %.4f" % (i+1, prec, recall)
		m.avg_prec += prec
		m.avg_recall += recall

	print "Precision: %.4f\nRecall: %.4f" % (
		m.avg_prec/mec.iters, 
		m.avg_recall/mec.iters
	)

if __name__ == '__main__':
	u = Util()
	args = u.cmdline_argparse()
	save_pickle = not args.load_pickle
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

		to_pickle = {
			'all_words': all_words,
			'init_weights': weights,
			'train_features': train_features,
			'train_labels': train_labels,
			'alg_problems': mec.alg_problems,
			'arith_problems': mec.arith_problems,
			'geo_problems': mec.geo_problems
		}

		mec.util.pickle_objs(prefix, to_pickle)
	
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
			args.n_steps,
			args.learn_rate,
			args.reg_coeff
		)

		mec.util.pickle_objs('./data/', {'train_weights': weights})
		
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

	print "\nTop 10 important features:"

	for k, v in sorted(mec.alg.iteritems(), key=lambda(k, v): (v, k), reverse=True)[:10]:
		print "\t%s:\t%f" % (k, v)
