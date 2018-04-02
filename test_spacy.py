#!/usr/bin/python

import re
import spacy
from spacy.symbols import ORTH, LEMMA, POS

sent = "From a cask of wine containing [M:L:F:M] gallons, [M:L:F:M] gallons were drawn. How much remained?"
nlp = spacy.load('en')
#nlp.tokenizer.add_special_case(u'[M:L:F:M]',
#	[
#		{
#			ORTH: u'[M:L:F:M]',
#			LEMMA: u'[M:L:F:M]',
#			POS: u'MATH'
#		}
#	]
#)

sents = nlp(sent.decode('utf8'))

indexes = [m.span() for m in re.finditer('\[M:.*?\]', sent)]

for start, end in indexes:
	sents.merge(start_idx=start, end_idx=end)
	print sents[start:end]

for token in sents:
	print('token.i: {2}\ttoken.idx: {0}\ttoken.pos: {3:10}token.text: {1}'.
		format(token.idx, token.text,token.i,token.pos_))

chunks = list(sents.noun_chunks)

print chunks
