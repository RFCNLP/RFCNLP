#!/usr/bin/env python3

import glob

tags = ["trigger", "action", "variable", 
		"transition", "error", "timer", 
	    "event*", "state", "event-lower", 
	    "event-upper"]


files = glob.glob("rfcs-annotated/*")
for file in files:
	print("Checking ... " + str(file))
	D = { t : (0, 0) for t in tags }
	with open(file, "r") as fr:
		for line in fr:
			for tag in tags:
				if not tag in line:
					continue
				(a, b) = D[tag]
				if "<" + tag in line:
					a += 1
				if "</" + tag in line:
					b += 1
				D[tag] = (a, b)
	print([(k, v) for k, v in D.items() if v[0] != v[1]])
	print("\n-----")


