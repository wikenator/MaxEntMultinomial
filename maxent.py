#!/usr/bin/python

import os, sys, pickle#, argparse
import numpy
from MaxEntClassifier import MaxEntClassifier as MEC
from FeatureBuilder import FeatureBuilder as FB
from Utils import Utils as Util

# Naive Bayes classification
def NB_Test(m, pkl, b, t):
	for i in xrange(mec.iters):
		m.split_sets(0.8, pkl)
		m.compute_base_probs(b, t)
		prec, recall = m.calculate_probs(b, t)

		print "Pass %d\n\tPrecision: %.4f\n\tRecall: %.4f" % (i+1, prec, recall)
		m.avg_prec += prec
		m.avg_recall += recall

	print "Precision: %.4f\nRecall: %.4f" % (
		m.avg_prec/mec.iters, 
		m.avg_recall/mec.iters
	)

def report(m, f, r, min_cost, best_learn_rate):
	print "\nTraining accuracies:", m.train_acc
	print "Test accuracies:", m.test_acc
	print "Precisions:", m.precision
	print "Recalls:", m.recall
	print "\nAverage training accuracy: %.5f" % (sum(m.train_acc)/args.folds)
	print "Average test accuracy: %.5f" % (sum(m.test_acc)/args.folds)
	print "Average precision: %.5f" % (sum(m.precision)/args.folds)
	print "Average recall: %.5f" % (sum(m.recall)/args.folds)

	if args.folds == 1:
		if not args.no_retrain:
			print "\nMaxEnt Stats:\n\tMinimized Cost: %.6f\n\tBest Learning Rate: %.6f" % (min_cost, best_learn_rate)

		print "\nTop 10 important features:"

		all_feats = m.alg
		all_feats.update(m.arith)
		all_feats.update(m.geo)

		for k, v in sorted(all_feats.iteritems(), key=lambda(k, v): (v, k), reverse=True)[:10]:
			print "\t%s:\t%f" % (k, v)

if __name__ == '__main__':
	util = Util()
	args = util.cmdline_argparse()
	prefix = './pickles/'

	mec = MEC(args.load_pickle)
	fb = FB(mec)

	# calculate train/test set split percentages
	if args.folds > 1: pct_split = 1 - (1.0 / args.folds)
	else: pct_split = 0.8

	# run Naive Bayes classifier
	if args.naive:
		mec.iters = 5
		NB_Test(mec, args.load_pickle, args.use_bigrams, args.use_trigrams)

	# perform k-fold cross-validation
	for fold in xrange(args.folds):
		sys.stderr.write("\nRunning fold " + str(fold+1) + "\n")

		mec.fold = fold
		mec.split_sets(pct_split, args.load_pickle)
		mec.compute_base_probs(args.use_bigrams, args.use_trigrams)
		mec.set_stats()

		# load previously saved data
		if args.load_pickle:
			sys.stderr.write("Reading pickle files.\n")

			words_pkl = open(prefix+'all_words'+str(fold)+'.pkl', 'rb')
			all_words = pickle.load(words_pkl)
			words_pkl.close()

			dep_pkl = open(prefix+'all_words'+str(fold)+'.pkl', 'rb')
			all_deps = pickle.load(dep_pkl)
			dep_pkl.close()

			wts_pkl = open(prefix+'init_weights'+str(fold)+'.pkl', 'rb')
			weights = pickle.load(wts_pkl)
			wts_pkl.close()

			feat_pkl = open(prefix+'train_features'+str(fold)+'.pkl', 'rb')
			train_features = pickle.load(feat_pkl)
			feat_pkl.close()

			label_pkl = open(prefix+'train_labels'+str(fold)+'.pkl', 'rb')
			train_labels = pickle.load(label_pkl)
			label_pkl.close()

		else:
			all_words = fb.get_vocabulary(args.use_bigrams, args.use_trigrams)

			if args.dep_parse:
				mec.all_deps = fb.get_dependencies(fb.all_problems)

			weights = fb.get_init_weights(all_words, args.use_bigrams, args.use_trigrams, args.dep_parse)
			train_features, train_labels = fb.get_train_features(all_words, args.use_bigrams, args.use_trigrams, args.dep_parse)

		if args.save_pickle:
			to_pickle = {
				'all_words': all_words,
#				'all_deps': all_deps,
				'init_weights': weights,
#				'train_features': train_features,
#				'train_labels': train_labels,
				'alg_problems': mec.alg_problems,
				'arith_problems': mec.arith_problems,
				'geo_problems': mec.geo_problems
			}

			mec.util.pickle_objs(prefix, fold, to_pickle)

		# run maxent classifier
		# default values: steps=1000
		#		  learn_rate=5e-4
		#		  reg_coeff=0.001
		#		  threshold=1e-5
		if args.no_retrain and os.path.exists('./data/train_weights'+str(fold)+'.pkl'):
			weights = pickle.load(open('./data/train_weights'+str(fold)+'.pkl', 'rb'))
			min_cost = 0
			best_learn_rate = 0

		else:
			weights, min_cost, best_learn_rate = mec.maxent(
				train_features, 
				weights, 
				train_labels, 
				args.steps,
				args.learn_rate,
				args.reg_coeff
			)

			if args.save_pickle:
				mec.util.pickle_objs('./data/', fold, {'train_weights': weights})
		
		# find training accuracy
		class_prob_train = numpy.dot(weights, train_features)
		class_bin_train = mec.hard_classify(class_prob_train)

		mec.train_acc.append(((class_bin_train == train_labels[0]).sum().astype(float)/len(class_bin_train)))

		err_report = open('train_misclassify.txt', 'a')
		err_report.write('FOLD ' + str(fold) + '\n')

		for i, l in enumerate(train_labels[0]):
			if class_bin_train[i] != l:
				err_report.write("%s#@#%s#@#%s\n" % (mec.util.categories[class_bin_train[i]], mec.train_problems[i][1], mec.train_problems[i][0]))

		err_report.close()

		# find testing accuracy
		test_features, test_labels = fb.get_test_features(all_words, args.use_bigrams, args.use_trigrams, args.dep_parse)
		class_prob_test = numpy.dot(weights, test_features)
		class_bin_test = mec.hard_classify(class_prob_test)

		confusion_matrix = mec.maxent_confusion_matrix(test_labels[0], class_bin_test)
		prec, rec = mec.precision_recall(confusion_matrix)
		mec.precision.append(prec)
		mec.recall.append(rec)

		mec.test_acc.append(((class_bin_test == test_labels[0]).sum().astype(float)/len(class_bin_test)))

		err_report = open('test_misclassify.txt', 'a')
		err_report.write('FOLD ' + str(fold) + '\n')

		for i, l in enumerate(test_labels[0]):
			if class_bin_test[i] != l:
				err_report.write("%s#@#%s#@#%s\n" % (mec.util.categories[class_bin_test[i]], mec.test_problems[i][1], mec.test_problems[i][0]))

		err_report.close()

		# free up memory for next iteration
		del all_words
		del weights
		del train_features
		del train_labels
		del test_features
		del test_labels

		if args.dep_parse: del mec.all_deps

	report(mec, args.folds, args.no_retrain, min_cost, best_learn_rate)
