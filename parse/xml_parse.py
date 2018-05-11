#!/usr/bin/python

import os, sys, re, glob
import pythonAPI as pyapi
import nltk
from nltk import RegexpTokenizer
import xml.etree.ElementTree as ET

if len(sys.argv) < 2:
	sys.exit('Please specify a folder to process')

else:
	subject = sys.argv[1]
	xml_files = glob.glob('./' + subject + '/*.xml')
#	regex_patterns = [(r'^\$\$?.*?\$\$?$', 'MATH')]
#	regex_tagger = nltk.RegexpTagger(regex_patterns)
	# tokenize words, latex expressions, and other non-space characters
	regex_tokenizer = RegexpTokenizer(r'\w+|\${1,2}(\\\$)?.*?\${1,2}|\S+')
	# regex for finding latex expressions
	latex_delim = re.compile(r'^\${1,2}((\\\$)?.*?)\${1,2}$')
	
	for xml_file in xml_files:
		tree = ET.parse(xml_file)
		root = tree.getroot()
		xml_file = os.path.splitext(os.path.basename(xml_file))[0]

		sys.stderr.write("Processing file: " + xml_file + '\n')

		with open('./' + subject + '_' + xml_file + '.txt', 'w') as fwrite:
			for i, q in enumerate(root.findall('problem/question')):
				sys.stderr.write("Question #" + str(i+1) + '\r')

				tokens = regex_tokenizer.tokenize(re.sub('\n', '', q.text))

				for i, t in enumerate(tokens):
					dollar = False

					if latex_delim.search(t):
						if re.match(r'\$\\\$', t): dollar = True

						tokens[i] = pyapi.detex(latex_delim.sub(r'\1', t))

						if dollar: tokens[i] = '$' + tokens[i]

#				print ' '.join(tokens) + '\n'
				fwrite.write(subject + '#@#' + ' '.join(tokens) + '\n')

			print

			fwrite.close()
