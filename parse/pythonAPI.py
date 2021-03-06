#!/usr/bin/python

import os, sys, re
from subprocess import Popen, PIPE

def detex(latexExpr):
        info = "parser#@!detex#@#problem#@!" + str(latexExpr) + "\n"
        p = Popen(['./parser_handler.pl'], stdin=PIPE, stdout=PIPE, stderr=PIPE)

        p.stdin.write(info)
        result = p.stdout.read()
        result = re.sub(r'""', '', result)

        return result

def abstract(latexExpr):
	info = "parser#@!abstract#@#problem#@!" + str(latexExpr) + "\n"
	p = Popen(['/home/arnold/git_repos/MaxEntMultinomial/parse/parser_handler.pl'], stdin=PIPE, stdout=PIPE, stderr=PIPE)

	p.stdin.write(info)
	result = p.stdout.read()
	result = re.sub(r'""', '', result)

	return result
