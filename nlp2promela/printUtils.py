'''
name       : printUtils.py
author     : [redacted]
authored   : 9 June 2020
updated    : 9 June 2020
description: provides pretty-print utils for nlp2promela
'''

class bcolors:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'


# A print utility that only prints if debug=True
def debugPrint(string, debug=False):
	if debug:
		print(f"{bcolors.WARNING}{string}{bcolors.ENDC}")

def printTransition(transition, delta1=15, delta2=20):
    (a, B, c) = transition
    BB = ";".join(B) if not (str(type(B)) == "<class 'str'>") else B
    
    prefix  = str(a)
    inner   = " ---"  + BB
    postfix = "---> " + str(c)

    prefix_to_inner_pad  = max(0, delta1 - len(prefix)) * " "
    inner_to_postfix_pad = max(0, delta2 - len(inner) ) * " "

    return prefix + prefix_to_inner_pad + inner + inner_to_postfix_pad + postfix

def deColor(str):
    return str.replace(bcolors.HEADER,    "")\
              .replace(bcolors.OKBLUE,    "")\
              .replace(bcolors.OKGREEN,   "")\
              .replace(bcolors.WARNING,   "")\
              .replace(bcolors.FAIL,      "")\
              .replace(bcolors.ENDC,      "")\
              .replace(bcolors.BOLD,      "")\
              .replace(bcolors.UNDERLINE, "")

def makeGreen(str):
    return f"{bcolors.OKGREEN}{str}{bcolors.ENDC}"

def makeRed(str):
    return f"{bcolors.WARNING}{str}{bcolors.ENDC}"

def makeBlue(str):
    return f"{bcolors.OKBLUE}{str}{bcolors.ENDC}"

def makeFail(str):
    return f"{bcolors.FAIL}{str}{bcolors.ENDC}"

def makeHeader(str):
    return f"{bcolors.HEADER}{str}{bcolors.ENDC}"

def printHeuristicRemoval(transition, heuristic, transitions=[], padwidth=60):
    transition = transition[0]
    prefix = "Removing " + printTransition(transition)
    padded = prefix + ( padwidth - len(prefix) ) * " "
    removalString = padded + " because of " + heuristic + " heuristic."
    if transitions == []:
        print(removalString)
    elif transition in transitions:
        print(makeRed(removalString))
    else:
        print(makeBlue(removalString))