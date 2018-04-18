#!/usr/bin/python

import os, sys, re, glob
import numpy
import nltk
import spacy
from Utils import Utils as Util

vocab = {}
sents = {}
pos_sents = {}
counts = []
alg_count = 0
arith_count = 0
geo_count = 0
ques_count = 0
u = Util()
files = glob.glob('./processed/*.txt')
files.sort()

nlp = spacy.load('en_core_web_lg', disable=['vectors', 'ner'])

for f in files:
	with open(f, 'r') as fh:
		f = os.path.splitext(os.path.basename(f))[0]
		sent_count = 0
		word_count = 0

		for sent in fh.read().splitlines():
			pid, cat, sent = sent.split('#@#')

			if cat == 'algebra': alg_count += 1
			elif cat == 'arithmetic': arith_count += 1
			elif cat == 'geometry': geo_count += 1

			if sent not in sents: sents[sent] = 0

			sents[sent] += 1
			ques_count += 1

			s_tok = nltk.tokenize.sent_tokenize(sent)
			w_tok = u.regex_tokenizer(sent)
			sent_count += len(s_tok)
			word_count += len(w_tok)

			for w in w_tok:
				if w not in vocab: vocab[w] = 1

				vocab[w] += 1

			doc = spacy.tokens.doc.Doc(nlp.vocab, words=[w.decode('utf8') for w in w_tok])

			for name, proc in nlp.pipeline: doc = proc(doc)

			pos_sent_arr = []

			for word in doc: pos_sent_arr.append(word.tag_)

			pos_sent = ' '. join(pos_sent_arr)
			
			if pos_sent not in pos_sents:
				pos_sents[pos_sent] = {
					'count': 0,
					'sents': []
				}

			pos_sents[pos_sent]['count'] += 1
			pos_sents[pos_sent]['sents'].append(sent)

		counts.append([sent_count, word_count])

		fh.close()

counts = numpy.matrix(counts)

print "Algebra: %d\nArithmetic: %d\nGeometry: %d" % (alg_count, arith_count, geo_count)
print '\nAvg sents/ques: ' + str(float(sum(counts[:, 0]))/ques_count)
print 'Avg words/sent: ' + str(float(sum(counts[:, 1]))/float(sum(counts[:, 0])))
print 'Avg words/ques: ' + str(float(sum(counts[:, 1]))/ques_count)
print '\nVocabulary size: ' + str(len(vocab))
print 'Data size: ' + str(sum(vocab.values()))
print '\nDuplicates (literal): %d' % (sum([int(v) for v in sents.values() if int(v) > 1]))

for k, v in sents.iteritems():
	if int(v) > 1: print "%d: %s" % (v, k)

print '\nDuplicates (POS): %d' % (sum([int(v['count']) for v in pos_sents.values() if int(v['count']) > 1]))

for pos_sent, sents in pos_sents.iteritems():
	if int(sents['count'] > 1):
		print "%d: %s" % (sents['count'], pos_sent)

		for s in sents['sents']:
			print "\t%s" % s
