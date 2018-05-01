#!/usr/bin/python

import os, sys, pickle
import numpy
from copy import deepcopy
from MaxEntClassifier import MaxEntClassifier as MEC
from FeatureBuilder import FeatureBuilder as FB
from Utils import Utils as Util

# Naive Bayes classification
def NB_Test(m, pkl, b, t):
	for i in xrange(mec.iters):
		m.split_sets(0.9, pkl)
		m.compute_base_probs(b, t)
		prec, recall = m.calculate_probs(b, t)

		print "Pass %d\n\tPrecision: %.4f\n\tRecall: %.4f" % (i+1, prec, recall)
		m.avg_prec += prec
		m.avg_recall += recall

	print "Precision: %.4f\nRecall: %.4f" % (
		m.avg_prec/mec.iters, 
		m.avg_recall/mec.iters
	)

# print statistics for training and test data
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

	# create MaxEnt Classifier object
	mec = MEC(args.load_pickle)
	# create FeatureBuilder object
	fb = FB(mec)

	# calculate train/test set split percentages
	if args.folds > 1: pct_split = 1 - (1.0 / args.folds)
	else: pct_split = 0.9

	# run Naive Bayes classifier
	if args.naive:
		mec.iters = 5
		NB_Test(mec, args.load_pickle, args.use_bigrams, args.use_trigrams)

	# perform k-fold cross-validation
	for fold in range(args.folds):
		sys.stdout.write("\nRunning fold " + str(fold+1) + "\n")

		mec.fold = fold
		mec.split_sets(pct_split, args.load_pickle)
		mec.compute_base_probs(args.use_bigrams, args.use_trigrams)
		mec.set_stats()

		# load previously saved data
		if args.load_pickle:
			sys.stdout.write("Reading pickle files.\n")

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

		elif args.grid_search:
			# train/validation set split
			validate_count = int(train_features.shape[1] / pct_split * (1 - pct_split))
			validate_features = train_features[:,:validate_count]
			validate_labels = train_labels[:,:validate_count]
			train_features = train_features[:,validate_count:]
			train_labels = train_labels[:,validate_count:]

			lr = [
				.9, .8, .7, .6, .5, .4, .3, .2, .1,
				.09, .08, .07, .06, .05, .04, .03, .02, .01,
				.009, .008, .007, .006, .005, .004, .003, .002, .001,
				9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4,
				9e-5, 8e-5, 7e-5, 6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5
			]
			rc = [
				.9, .8, .7, .6, .5, .4, .3, .2, .1,
				.09, .08, .07, .06, .05, .04, .03, .02, .01,
				.009, .008, .007, .006, .005, .004, .003, .002, .001,
				9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4,
				9e-5, 8e-5, 7e-5, 6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5
			]
			max_acc = 0.0
			max_v_acc = 0.0
			best_lr = 0.0
			best_rc = 0.0
			init_weights = deepcopy(weights)
			best_wts = deepcopy(init_weights)

			for l in lr:
				for r in rc:
					sys.stdout.write("Learn Rate: %.5f, Reg Coeff: %.5f\n" % (l, r))

					weights, min_cost, best_learn_rate = mec.maxent_grid_search(
						train_features, 
						validate_features,
						deepcopy(init_weights), 
						train_labels, 
						validate_labels, 
						args.steps,
						l,
						r
					)

					if min_cost == 'div' or min_cost == 'no': continue

					class_prob_train = numpy.dot(weights, train_features)
					class_bin_train = mec.hard_classify(class_prob_train)

					t_acc = ((class_bin_train == train_labels[0]).sum().astype(float)/len(class_bin_train))

					class_val_train = numpy.dot(weights, validate_features)
					class_bin_val = mec.hard_classify(class_val_train)

					v_acc = ((class_bin_val == validate_labels[0]).sum().astype(float)/len(class_bin_val))

					if t_acc >= max_acc:
						max_acc = t_acc

						if v_acc > max_v_acc:
							max_v_acc = v_acc
							best_lr = l
							best_rc = r
							best_wts = deepcopy(weights)

			weights = best_wts

			print "best learn rate: %.5f, best reg coeff: %.5f" % (best_lr, best_rc)

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

#		err_report = open('train_misclassify.txt', 'a')
#		err_report.write('FOLD ' + str(fold) + '\n')

#		for i, l in enumerate(train_labels[0]):
#			if class_bin_train[i] != l:
#				err_report.write("%s#@#%s#@#%s\n" % (mec.util.categories[class_bin_train[i]], mec.train_problems[i][1], mec.train_problems[i][0]))

#		err_report.close()

		# find testing accuracy
		test_features, test_labels = fb.get_test_features(all_words, args.use_bigrams, args.use_trigrams, args.dep_parse)
		class_prob_test = numpy.dot(weights, test_features)
		class_bin_test = mec.hard_classify(class_prob_test)

		# create confusion matrix for precision and recall calculations
		confusion_matrix = mec.maxent_confusion_matrix(test_labels[0], class_bin_test)
		prec, rec = mec.precision_recall(confusion_matrix)
		mec.precision.append(prec)
		mec.recall.append(rec)

		mec.test_acc.append(((class_bin_test == test_labels[0]).sum().astype(float)/len(class_bin_test)))

#		err_report = open('test_misclassify.txt', 'a')
#		err_report.write('FOLD ' + str(fold) + '\n')

#		for i, l in enumerate(test_labels[0]):
#			if class_bin_test[i] != l:
#				err_report.write("%s#@#%s#@#%s\n" % (mec.util.categories[class_bin_test[i]], mec.test_problems[i][1], mec.test_problems[i][0]))

#		err_report.close()

		# free up memory for next iteration
		del all_words
		del weights
		del train_features
		del train_labels
		del test_features
		del test_labels

		if args.dep_parse: del mec.all_deps

	report(mec, args.folds, args.no_retrain, min_cost, best_learn_rate)
