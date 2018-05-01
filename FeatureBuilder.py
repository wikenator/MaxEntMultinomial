#!/usr/bin/python

'''
Author: Arnold Wikey
Date: 2018
Description: class containing functions for building feature sets for MaxEnt classification. Feature sets include unigrams, bigrams, trigrams, and dependency parses.
'''

import sys, re
import math, numpy
import spacy
from nltk.util import ngrams

class FeatureBuilder():
	def __init__(self, mec):
		self.mec = mec
		self.nlp = spacy.load('en_core_web_lg', disable=['vectors', 'ner'])
		
		self.train_acc = []
		self.test_acc = []
		self.precision = []
		self.recall = []

		self.all_problems = self.mec.alg_problems + self.mec.arith_problems + self.mec.geo_problems

	# add all word tokens into a set and sort tokens alphabetically
	def get_vocabulary(self, use_bigrams, use_trigrams):
		sys.stdout.write("Compiling vocabulary.\n")

		vocab = [w for p in self.mec.train_problems for w in self.mec.util.regex_tokenizer(p[0].lower())]

		if use_bigrams:
			vocab += [b for p in self.mec.train_problems for b in ngrams(self.mec.util.regex_tokenizer(p[0].lower()), 2)]

		if use_trigrams:
			vocab += [t for p in self.mec.train_problems for t in ngrams(self.mec.util.regex_tokenizer(p[0].lower()), 3)]

		vocab = set(vocab)
		vocab = list(vocab)
		vocab.sort()

		return vocab

	# convert text category to integer index
	def cat_to_int(self, category):
		if category == 'algebra': return 0
		elif category == 'arithmetic': return 1
		elif category == 'geometry': return 2

	# create dependency triple based on current word token from spacy Doc object
	def create_dependency_trigram(self, word, stl_idx):
		if word.i == 0:
			prev_tok_tag = 'null'
			next_tok_tag = word.nbor().tag_

		elif word.i == stl_idx:
			prev_tok_tag = word.nbor(-1).tag_
			next_tok_tag = 'null'

		else:
			prev_tok_tag = word.nbor(-1).tag_
			next_tok_tag = word.nbor().tag_

		return (prev_tok_tag, word.tag_, next_tok_tag)

	# gather all dependency triples to be used in feature space
	def get_dependencies(self, sets):
		sys.stdout.write('Compiling dependencies.\n')

		dep_trigrams = {}
		dep_feats = []
		alg_counts = {}
		arith_counts = {}
		geo_counts = {}

		for i, (p, c) in enumerate(sets):
			doc = spacy.tokens.doc.Doc(self.nlp.vocab, words=self.mec.util.regex_tokenizer(p.decode('utf8'), True))

			for name, proc in self.nlp.pipeline: doc = proc(doc)

			for word in doc:
				# only interested in dependency relation involving math tokens
				if re.match('\[M:', word.text):
					if word.i == 0:
						prev_tok_txt = 'null'
						next_tok_txt = word.nbor().lower_

					elif word.i == len(doc)-1:
						prev_tok_txt = word.nbor(-1).lower_
						next_tok_txt = 'null'

					else:
						prev_tok_txt = word.nbor(-1).lower_
						next_tok_txt = word.nbor().lower_

					sym_tri = self.create_dependency_trigram(word, len(doc)-1)
					spec_tri = (
						c,
						prev_tok_txt,
						word.text,
						next_tok_txt
					)

					# restrict dependency triples to only contain verb parts of speech
					if any(any(verb_pos in s for s in sym_tri) for verb_pos in ['VB', 'MD']):
						if sym_tri not in dep_trigrams:
							dep_trigrams[sym_tri] = {}

						if spec_tri not in dep_trigrams[sym_tri]:
							dep_trigrams[sym_tri][spec_tri] = {
								'count': 0,
								'deps': {}
							}

						dep_trigrams[sym_tri][spec_tri]['count'] += 1

						if word.head.i == word.i:
							dep_txt = 'null'

						else:
							dep = word.head
							dep_txt = dep.lower_

						dep = (word.text, word.dep_, dep_txt)

						if dep not in dep_trigrams[sym_tri][spec_tri]['deps']: dep_trigrams[sym_tri][spec_tri]['deps'][dep] = 0

						dep_trigrams[sym_tri][spec_tri]['deps'][dep] += 1

						if c == 'algebra':
							if sym_tri not in alg_counts: alg_counts[sym_tri] = 1.0

							alg_counts[sym_tri] += 1

						elif c == 'arithmetic':
							if sym_tri not in arith_counts: arith_counts[sym_tri] = 1.0

							arith_counts[sym_tri] += 1

						elif c == 'geometry':
							if sym_tri not in geo_counts: geo_counts[sym_tri] = 1.0

							geo_counts[sym_tri] += 1

						# append dependency triple to dependency features list
						dep_feats.append(dep)

		# restrict dependency triples to single occurrences
##		for i, (sym_t, spec_t) in enumerate(dep_trigrams.iteritems()):
##			tri_cat_sum = {c: len([k for k in spec_t.keys() if k[0] == c]) for c in self.mec.util.categories}
##			dep_cats = [k for k, v in tri_cat_sum.iteritems() if v]

##			if len(dep_cats) == 1:
##				c = dep_cats[0]
##				deps = [d for k, v in spec_t.iteritems() for d in v['deps'] if k[0] == c]

##				for d in deps: dep_feats.append(d)

		alg_counts[('<UNK>', '<UNK>', '<UNK>')] = 1.0
		arith_counts[('<UNK>', '<UNK>', '<UNK>')] = 1.0
		geo_counts[('<UNK>', '<UNK>', '<UNK>')] = 1.0

		self.mec.alg_dep = alg_counts
		self.mec.arith_dep = arith_counts
		self.mec.geo_dep = geo_counts

		total_alg = float(sum(self.mec.alg_dep.values()))
		total_arith = float(sum(self.mec.arith_dep.values()))
		total_geo = float(sum(self.mec.geo_dep.values()))

		for d in self.mec.alg_dep: self.mec.alg_dep[d] /= total_alg
		for d in self.mec.arith_dep: self.mec.arith_dep[d] /= total_arith
		for d in self.mec.geo_dep: self.mec.geo_dep[d] /= total_geo

		# sort features and remove duplicates
		dep_feats = set(dep_feats)
		dep_feats = list(dep_feats)
		dep_feats.sort()

		return dep_feats

	# abstract for get_weights
	def get_init_weights(self, V, use_bigrams, use_trigrams, dep_parse):
		sys.stdout.write("Initializing weights.\n")

		# set weights with dependency parse features
		if dep_parse:
			D = self.mec.all_deps
			wts = numpy.zeros((3, len(V)+len(D)))

			# process algebra problem set
			wts = self.get_weights(
				self.mec.alg_train_set, self.mec.alg,
				V, wts, 0, 
				use_bigrams, use_trigrams,
				dep_parse, D, self.mec.alg_dep
			)

			# process arithmetic problem set
			wts = self.get_weights(
				self.mec.arith_train_set, self.mec.arith,
				V, wts, 1, 
				use_bigrams, use_trigrams, 
				dep_parse, D, self.mec.arith_dep
			)

			# process geometry problem set
			wts = self.get_weights(
				self.mec.geo_train_set, self.mec.geo,
				V, wts, 2, 
				use_bigrams, use_trigrams, 
				dep_parse, D, self.mec.geo_dep
			)

		# set weights with just vocabulary features
		else:
			wts = numpy.zeros((3, len(V)))

			# process algebra problem set
			wts = self.get_weights(
				self.mec.alg_train_set, self.mec.alg,
				V, wts, 0, 
				use_bigrams, use_trigrams
			)

			# process arithmetic problem set
			wts = self.get_weights(
				self.mec.arith_train_set, self.mec.arith,
				V, wts, 1, 
				use_bigrams, use_trigrams
			)

			# process geometry problem set
			wts = self.get_weights(
				self.mec.geo_train_set, self.mec.geo,
				V, wts, 2, 
				use_bigrams, use_trigrams
			)

		return wts
		
	# set initial weights to raw count probabilities
	def get_weights(self, t_set, probs, V, wts, idx, b, t, d = False, D = None, dep_probs = None):
		for p, c in t_set:
			tokens = self.mec.util.regex_tokenizer(p.lower())

			for w in tokens:
				if w in V: wts[idx][V.index(w)] = probs[w]

			if b:
				bigrams = ngrams(tokens, 2)

				for b in bigrams:
					if b in V: wts[idx][V.index(b)] = probs[b]

			if t:
				trigrams = ngrams(tokens, 3)

				for t in trigrams:
					if t in V: wts[idx][V.index(t)] = probs[t]

			if d:
				doc = spacy.tokens.doc.Doc(self.nlp.vocab, words=self.mec.util.regex_tokenizer(p.decode('utf8'), True))

				for name, proc in self.nlp.pipeline:
					doc = proc(doc)

				for word in doc:
					if re.match('\[M:', word.text):
						sym_tri = self.create_dependency_trigram(word, len(doc)-1)

						# restrict dependency triple to only contain verb parts of speech
						if any(any(verb_pos in s for s in sym_tri) for verb_pos in ['VB', 'MD']):
							if word.head.i == word.i:
								dep_txt = 'null'

							else:
								dep = word.head
								dep_txt = dep.lower_

							dep = (word.text, word.dep_, dep_txt)

							# set weight for dependency triple to be probability of symbolic representation of triple
							if dep in D: wts[idx][D.index(dep)] = dep_probs[sym_tri]

		return wts	

	# abstract for get_features
	def get_train_features(self, V, use_bigrams, use_trigrams, dep_parse):
		sys.stdout.write("\nVectorizing training features.\n")

		sets = self.mec.train_problems

		if dep_parse: feats = numpy.zeros((len(V)+len(self.mec.all_deps), len(sets)))
		else: feats = numpy.zeros((len(V), len(sets)))

		labels = numpy.empty((1, len(sets)), dtype=int)

		for i, (p, c) in enumerate(sets):
			sys.stderr.write("Getting features for item "+str(i+1)+'\r')

			feats = self.get_word_features(i, feats, V, p, use_bigrams, use_trigrams)

			if dep_parse: feats = self.get_dependency_features(i, feats, self.mec.all_deps, p)

			labels[0][i] = self.cat_to_int(c)

		return feats, numpy.array(labels)

	# abstract for get_features
	def get_test_features(self, V, use_bigrams, use_trigrams, dep_parse):
		sys.stdout.write("\nVectorizing test features.\n")

		sets = self.mec.test_problems

		if dep_parse: feats = numpy.zeros((len(V)+len(self.mec.all_deps), len(sets)))
		else: feats = numpy.zeros((len(V), len(sets)))

		labels = numpy.empty((1, len(sets)), dtype=int)

		for i, (p, c) in enumerate(sets):
			sys.stderr.write("Getting features for item "+str(i+1)+'\r')

			feats = self.get_word_features(i, feats, V, p, use_bigrams, use_trigrams)

			if dep_parse: feats = self.get_dependency_features(i, feats, self.mec.all_deps, p)

			labels[0][i] = self.cat_to_int(c)

		return feats, numpy.array(labels)

	# convert word tokens into 0-1 features
	def get_word_features(self, i, feats, V, p, use_bigrams, use_trigrams):
		tokens = self.mec.util.regex_tokenizer(p.lower())

		for w in tokens:
			if w in V: feats[V.index(w)][i] = 1

		if use_bigrams:
			bigrams = ngrams(tokens, 2)

			for b in bigrams:
				if b in V: feats[V.index(b)][i] = 1

		if use_trigrams:
			trigrams = ngrams(tokens, 3)

			for t in trigrams:
				for t in V: feats[V.index(t)][i] = 1

		return feats

	# convert dependency triples into 0-1 features
	def get_dependency_features(self, i, feats, D, p):
		# regex tokenize problem and create spacy Doc object
		doc = spacy.tokens.doc.Doc(self.nlp.vocab, words=self.mec.util.regex_tokenizer(p.decode('utf8'), True))

		for name, proc in self.nlp.pipeline:
			doc = proc(doc)

		for word in doc:
			# only find word tokens that contain a math object
			if re.match('\[M:', word.text):
				sym_tri = self.create_dependency_trigram(word, len(doc)-1)

				# restrict dependency triples to only contain verb parts of speech
				if any(any(verb_pos in s for s in sym_tri) for verb_pos in ['VB', 'MD']):
					if word.head.i == word.i:
						dep_txt = 'null'

					else:
						dep = word.head
						dep_txt = dep.lower_

					dep = (word.text, word.dep_, dep_txt)

					# activate dependency triple feature for this problem
					if dep in D: feats[D.index(dep)][i] = 1

		return feats
