'''
name       : stringUtils.py
author     : [redacted]
authored   : 9 June 2020
updated    : 9 June 2020
description: provides string parsing utils for nlp2promela
'''
from printUtils import debugPrint

import re

# INPUT:
# 	string  - a string which might have a substring in options
# 	options - a list of possible substrings
# OUTPUT:
#	bst - an opt in options that spans a maximal portion of string
def bestGuess(string, options):
	debugPrint("Guessing for: " + str(string))
	if string == None:
		return string
	k = 0
	bst = string
	for opt in options:
		debugPrint("\tConsidering: " + opt)
		if opt in string and len(opt) > k:
			debugPrint("\tSetting to: " + opt)
			bst = opt
			k = len(opt)
	if not bst in options:
		return None
	return bst

# https://stackoverflow.com/a/1267145/1586231
# Simple utility function to check if a string represents an int
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

# Cleans up a string to be Promela-compliant
def cleanUp(text):
	return text.replace("an ",     "" )\
	           .replace("a ",      "" )\
	           .replace("segment", "" )\
	           .replace(" ",       "_")\
	           .replace("\t",      "_")\
	           .replace("\f",      "_")\
	           .replace("-",       "_")\
	           .strip()

def cleanFile(file):
	filename, extension = file.rsplit(".", 1)
	newfile = filename + "-clean." + extension
	removedlines = []
	i = 0
	with open(file, "r") as fr:
		with open(newfile, "w") as fw:
			for line in fr:
				l = line.replace("\n", "").strip().replace("\f", "")
				if l:
					fw.write(l + "\n")
				else:
					removedlines.append(i)
				i += 1
	return newfile, removedlines