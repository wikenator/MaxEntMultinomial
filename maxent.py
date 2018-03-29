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

def report(m, f, r, min_cost, best_learn_rate):
	print "\nTraining accuracies: ", m.train_acc
	print "Test accuracies: ", m.test_acc
	print "Precisions: ", m.precision
	print "Recalls: ", m.recall
	print "\nAverage training accuracy: %.5f" % (sum(m.train_acc)/args.folds)
	print "Average test accuracy: %.5f" % (sum(m.test_acc)/args.folds)
	print "Average precision: %.5f" % (sum(m.precision)/args.folds)
	print "Average recall: %.5f" % (sum(m.recall)/args.folds)

	if args.folds == 1:
		if not args.no_retrain:
			print "\nMaxEnt Stats:\n\tMinimized Cost: %.6f\n\tBest Learning Rate: %.6f" % (min_cost, best_learn_rate)

		print "\nTop 10 important features:"

		for k, v in sorted(m.alg.iteritems(), key=lambda(k, v): (v, k), reverse=True)[:10]:
			print "\t%s:\t%f" % (k, v)

if __name__ == '__main__':
	u = Util()
	args = u.cmdline_argparse()
	save_pickle = not args.load_pickle
	prefix = './pickles/'

	mec = MEC(save_pickle)
	k = 100

	# run Naive Bayes classifier
	if args.naive:
		mec.iters = 5
		NB_Test(mec, args.use_bigrams, args.use_trigrams)

	for fold in xrange(args.folds):
		sys.stderr.write("Running fold "+str(fold)+"\n")

		mec.fold = fold
		mec.split_sets(0.8, save_pickle)
		mec.compute_base_probs(k, args.use_bigrams, args.use_trigrams)
		mec.set_stats()

		if save_pickle or not os.path.exists('./pickles/all_words'+str(fold)+'.pkl'):
			if not save_pickle:
				sys.stderr.write("Creating new data.\n")

			all_words = mec.get_vocabulary(args.use_bigrams, args.use_trigrams)
			weights = mec.get_init_weights(all_words, args.use_bigrams, args.use_trigrams)
			train_features, train_labels = mec.get_train_features(all_words, args.use_bigrams, args.use_trigrams)
			to_pickle = {
				'all_words': all_words,
				'init_weights': weights,
				'train_features': train_features,
				'train_labels': train_labels,
				'alg_problems': mec.alg_problems,
				'arith_problems': mec.arith_problems,
				'geo_problems': mec.geo_problems
			}

			mec.util.pickle_objs(prefix, fold, to_pickle)

		else:
			sys.stderr.write("Reading pickle files.\n")

			all_words = pickle.load(open(prefix+'all_words'+str(fold)+'.pkl', 'rb'))
			weights = pickle.load(open(prefix+'init_weights'+str(fold)+'.pkl', 'rb'))
			train_features = pickle.load(open(prefix+'train_features'+str(fold)+'.pkl', 'rb'))
			train_labels = pickle.load(open(prefix+'train_labels'+str(fold)+'.pkl', 'rb'))

		if args.no_retrain:
			weights = pickle.load(open('./data/train_weights'+str(fold)+'.pkl', 'rb'))
			min_cost = 0
			best_learn_rate = 0

		# run maxent classifier
		# default values: steps=1000
		#		  learn_rate=5e-4
		#		  reg_coeff=0.001
		#		  threshold=1e-5
		else:
			weights, min_cost, best_learn_rate = mec.maxent(
				train_features, 
				weights, 
				train_labels, 
				args.steps,
				args.learn_rate,
				args.reg_coeff
			)

			mec.util.pickle_objs('./data/', fold, {'train_weights': weights})
		
		class_prob_train = numpy.dot(weights, train_features)
		class_bin_train = mec.hard_classify(class_prob_train)

		mec.train_acc.append(((class_bin_train == train_labels).sum().astype(float)/len(class_bin_train)))

		test_features, test_labels = mec.get_test_features(all_words, args.use_bigrams, args.use_trigrams)
		class_prob_test = numpy.dot(weights, test_features)
		class_bin_test = mec.hard_classify(class_prob_test)

		confusion_matrix = mec.maxent_confusion_matrix(test_labels[0], class_bin_test)
		prec, rec = mec.precision_recall(confusion_matrix)
		mec.precision.append(prec)
		mec.recall.append(rec)

		mec.test_acc.append(((class_bin_test == test_labels).sum().astype(float)/len(class_bin_test)))
	# end for

	report(mec, args.folds, args.no_retrain, min_cost, best_learn_rate)
