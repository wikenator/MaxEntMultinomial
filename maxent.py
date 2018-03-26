#!/usr/bin/python

import sys, pickle, argparse
import numpy
from MaxEntClassifier import MaxEntClassifier as MEC

def NB_Test(m):
	for i in range(0, 5):
		m.split_sets(0.8)
		prec, recall = m.calculate_probs()

		print "Pass %d\n\tPrecision: %.4f\n\tRecall: %.4f" % (i+1, prec, recall)
		m.avg_prec += prec
		m.avg_recall += recall

	print "Precision: %.4f\nRecall: %.4f" % (
		m.avg_prec/mec.iters, 
		m.avg_recall/mec.iters
	)

def pickle_objs(m, prefix, v, w, f, l):
	pkl_vocab = open(prefix+'all_words.pkl', 'wb')
	pickle.dump(v, pkl_vocab)
	pkl_vocab.close()
	
	pkl_wts = open(prefix+'weights.pkl', 'wb')
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
#	pkl_group = cmd_line_parser.add_mutually_exclusive_group(required=False)
	cmd_line_parser.add_argument('-l', '--load_pickle', action='store_true', help='Load data from pickle files (will not recalculate data).')
#	pkl_group.add_argument('-s', '--save_pickle', action='store_true', help='Save data to pickle files after calculation.')
	args = cmd_line_parser.parse_args()
	
#	if (args.save_pickle == False) and (args.load_pickle == False):
#		args.save_pickle == True

	save_pickle = not args.load_pickle
	prefix = './pickles/'

	mec = MEC(args.load_pickle)
	mec.iters = 5

	if len(sys.argv) > 1 and sys.argv[1] == 'nb':
		NB_Test(mec)

	mec.split_sets(0.8, args.load_pickle)
	mec.set_stats()

	if not args.load_pickle:
		all_words = mec.get_vocabulary()
		weights = mec.get_init_weights(all_words, args.use_bigrams)
		train_features, train_labels = mec.get_train_features(all_words, args.use_bigrams)
	
	else:
		sys.stderr.write("Reading pickle files.\n")

		all_words = pickle.load(open(prefix+'all_words.pkl', 'rb'))
		weights = pickle.load(open(prefix+'weights.pkl', 'rb'))
		train_features = pickle.load(open(prefix+'train_features.pkl', 'rb'))
		train_labels = pickle.load(open(prefix+'train_labels.pkl', 'rb'))

	if save_pickle:
		pickle_objs(
			mec, 
			prefix, 
			all_words, 
			weights, 
			train_features, 
			train_labels
		)

	# run maxent classifier
	# default values: n_steps=1000
	#		  learn_rate=5e-4
	#		  reg_coeff=0.001
	#		  threshold=1e-4
	weights, min_cost, best_learn_rate = mec.maxent(
		train_features, 
		weights, 
		train_labels, 
		400, 
		5e-3, 
		0.02
	)
	class_prob_train = numpy.dot(weights, train_features)
	class_bin_train = mec.hard_classify(class_prob_train)

	print "\nTraining Accuracy: %.4f" % \
		((class_bin_train == train_labels).sum().astype(float)/len(class_bin_train))

	test_features, test_labels = mec.get_test_features(all_words, args.use_bigrams)
	class_prob_test = numpy.dot(weights, test_features)
	class_bin_test = mec.hard_classify(class_prob_test)

	print "\nTest Accuracy: %.4f" % \
		((class_bin_test == test_labels) \
		.sum() \
		.astype(float) / \
		len(class_bin_test))
	print "MaxEnt Stats:\n\tMinimized Cost: %.6f\n\tBest Learning Rate: %.6f" % (min_cost, best_learn_rate)
