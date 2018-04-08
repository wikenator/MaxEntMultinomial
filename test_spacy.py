#!/usr/bin/python

import sys, re
import glob
import spacy
from spacy.symbols import ORTH, LEMMA, POS
from Utils import Utils as Util

def showTree(sent):
	def __showTree(token, level):
		tab = "\t" * level
		sys.stdout.write("\n%s{" % (tab))
		[__showTree(t, level+1) for t in token.lefts]
		sys.stdout.write("\n%s\t%s [%s] (%s)" % (tab,token,token.dep_,token.tag_))
		[__showTree(t, level+1) for t in token.rights]
		sys.stdout.write("\n%s}" % (tab))
	return __showTree(sent.root, 1)

#s = "From a cask of wine containing [M:L:F:M] gallons , [M:L:F:M] gallons were drawn . How much remained ?"
#s = "If [M:L:N] cu. in. of gold beaten into gold leaf will cover [M:L:N] sq. in. of surface , find the thickness of the leaf .(Please give an exact answer in inches , in decimal form .)"

u = Util()

math_trigrams = {}
#nlp = spacy.load('en_core_web_lg', disable=['vectors', 'ner'])
nlp = spacy.load('en', disable=['vectors', 'ner'])

file_list = glob.glob('./processed/*.txt')
m_vb_tri = open('math_verb_trigrams.txt', 'w')

#file_list = [file_list[1]]
for f_name in file_list:
	f = open(f_name, 'r').readlines()

	for l in f:
		pid, c, p = l.strip().split('#@#')
	#	doc = nlp(p.decode('utf8'))
		doc = spacy.tokens.doc.Doc(nlp.vocab, words=u.regex_tokenizer(p.decode('utf8'), True))

		for name, proc in nlp.pipeline:
			doc = proc(doc)

		#for token in doc:
		#	if re.match('\[M:', str(token)):
			#	token.tag_ = 'MATH'
			#	token.pos_ = 'X'

#			print token.tag_, token.lower_, token.dep_, token.head.lower_

		#doc = nlp(p.decode('utf8'))
		sys.stderr.write('Processing problem #' + str(pid) + '\r')

		#indexes = [m.span() for m in re.finditer('\[M:.*?\]', p.decode('utf8'))]

		#for start, end in indexes:
		#	doc.merge(start, end, tag='MATH', pos='X')

		#for token in doc:
		#	print token.tag_, token.lower_, token.dep_, token.head.lower_

		#for sent in doc.sents:
		#	for word in sent:
		for word in doc:
			if re.match('\[M:', word.text):
			#if word.tag_ == 'MATH':
				if word.i == 0:
					prev_tok_tag = 'null'
					prev_tok_dep = 'null'
					prev_tok_txt = 'null'
					next_tok_tag = word.nbor().tag_
					next_tok_dep = word.nbor().dep_
					next_tok_txt = word.nbor().lower_

				elif word.i == len(doc)-1:
					prev_tok_tag = word.nbor(-1).tag_
					prev_tok_dep = word.nbor(-1).dep_
					prev_tok_txt = word.nbor(-1).lower_
					next_tok_tag = 'null'
					next_tok_dep = 'null'
					next_tok_txt = 'null'

				else:
					prev_tok_tag = word.nbor(-1).tag_
					prev_tok_dep = word.nbor(-1).dep_
					prev_tok_txt = word.nbor(-1).lower_
					next_tok_tag = word.nbor().tag_
					next_tok_dep = word.nbor().dep_
					next_tok_txt = word.nbor().lower_

				sym_tri = (
					prev_tok_tag,#+'#'+prev_tok_dep, 
					word.tag_,#+'#'+word.dep_, 
					next_tok_tag#+'#'+next_tok_dep
				)

				spec_tri = (
					c,
					prev_tok_txt,
					word.text,
					next_tok_txt
				)

				if sym_tri not in math_trigrams:
					math_trigrams[sym_tri] = {}

				if spec_tri not in math_trigrams[sym_tri]:
					math_trigrams[sym_tri][spec_tri] = {
						'count': 0,
						'dependency': {}
					}

#				math_trigrams[trigram] += 1
				math_trigrams[sym_tri][spec_tri]['count'] += 1

				if word.head.i == word.i:
				#	head_idx = 0
					dep_txt = 'null'
#					dep_tag = 'null'

				else:
					#head_idx = word.i-sent[0].i+1
					#head_idx = word.head.i
#					dep = sent[head_idx]
					dep = word.head
					dep_txt = dep.lower_
#					dep_tag = dep.tag_

				dep = (word.text, word.dep_, dep_txt)

				if dep not in math_trigrams[sym_tri][spec_tri]['dependency']: math_trigrams[sym_tri][spec_tri]['dependency'][dep] = 0

				math_trigrams[sym_tri][spec_tri]['dependency'][dep] += 1

sys.stderr.write('\n')

for sym_t, spec_t in math_trigrams.iteritems():
	#print str(sum(spec_t['trigram'].values())) + ': ' + str(sym_t)
	if any(any(verb_pos in s for s in sym_t) for verb_pos in ['VB', 'MD']):
#		print sym_t
		m_vb_tri.write(str(sym_t) + '\n')

#		tri_sum = 0

#		for t in spec_t:
#			tri_sum += int(spec_t[t]['count'])

#		if tri_sum > 0:
		for t in spec_t:
			#print '\t' + str(spec_t[t]['count']) + ':' + str(t)
			m_vb_tri.write('\t' + str(spec_t[t]['count']) + ':' + str(t) + '\n')

			for d in spec_t[t]['dependency']:
				#print '\t\t' + str(spec_t[t]['dependency'][d]) + ': ' + str(d)
				m_vb_tri.write('\t\t' + str(spec_t[t]['dependency'][d]) + ': ' + str(d) + '\n')

#[showTree(sent) for sent in doc.sents]
#for token in doc:
#	print('token.i: {2}\ttoken.pos: {3}\ttoken.tag: {4}\ttoken.text: {1}\ttoken.dep: {0}'.
#		format(token.dep_, token.text, token.i, token.pos_, token.tag_))

#for chunk in doc.noun_chunks:
#	print('c.text: {0}\tc.root.txt: {1}\tc.root.dep: {2}'.
#		format(chunk.text, chunk.root.text, chunk.root.dep_))
