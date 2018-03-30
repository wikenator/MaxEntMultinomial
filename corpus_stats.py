#!/usr/bin/python

import os, sys, re, glob
import numpy
import nltk
from Utils import Utils as Util

counts = []
alg_count = 0
arith_count = 0
geo_count = 0
ques_count = 0
u = Util()
files = glob.glob('./processed/*.txt')
files.sort()

for f in files:
	with open(f, 'r') as fh:
		f = os.path.splitext(os.path.basename(f))[0]
		sent_count = 0
		word_count = 0
#		counts[f] = {
#			's': 0,
#			'w': 0
#		}

#		sys.stderr.write('Processing file: ' + f + '\n')

		for sent in fh.read().splitlines():
			pid, cat, sent = sent.split('#@#')

			if cat == 'algebra': alg_count += 1
			elif cat == 'arithmetic': arith_count += 1
			elif cat == 'geometry': geo_count += 1

			ques_count += 1

			s_tok = nltk.tokenize.sent_tokenize(sent)
#			w_tok = nltk.tokenize.word_tokenize(sent)
			w_tok = u.regex_tokenizer(sent)
#			counts[f]['s'] += len(s_tok)
#			counts[f]['w'] += len(w_tok)
			sent_count += len(s_tok)
			word_count += len(w_tok)

		counts.append([sent_count, word_count])

		fh.close()

#for f in sorted(counts):
#	print f + '\n\tsentence counts: ' + str(counts[f]['s']) + '\n\tword counts: ' + str(counts[f]['w'])

#for i, c in enumerate(counts):
#	print files[i] + ':\n\tsentence counts: ' + str(c[0]) + '\n\tword counts: ' + str(c[1])

counts = numpy.matrix(counts)

print "Algebra: %d\nArithmetic: %d\nGeometry: %d" % (alg_count, arith_count, geo_count)
print '\nAvg sents/ques: ' + str(float(sum(counts[:, 0]))/ques_count)
print 'Avg words/sent: ' + str(float(sum(counts[:, 1]))/float(sum(counts[:, 0])))
