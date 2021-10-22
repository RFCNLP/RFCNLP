'''
name       : fsmUtils.py
author     : [redacted]
authored   : 9 June 2020
updated    : 9 June 2020
description: provides finite state machine class for nlp2promela
'''
from transitions.extensions import GraphMachine as Machine
from networkx.drawing.nx_agraph import to_agraph

import matplotlib.pyplot as plt
import os.path

from graphUtils import printSummaryOfDifference, showgraph
from printUtils import debugPrint

# Class we use based on transitions package to represent FSMs
class FSM:

    def __init__(self, 
                 states, 
                 s0, 
                 transitions, 
                 labeled=True, 
                 removedLines=None,
                 msgs=None):

        self.removedLines = removedLines

        if labeled:

	        T = []
	        L = {}

	        for ((a, B, c), l) in list(transitions):

	        	tr = (a, ";".join(B), c)
	        	if not tr in T:
	        		T.append(tr)

	        	if not tr in L:
	        		L[tr] = [ l ]
	        	else:
	        		L[tr].append(l)

        	self.transitions = T
        	self.labelmap    = L

        else:

        	self.transitions = [(a, ";".join(B), c) \
                                for (a, B, c) in transitions]

        	self.labelmap    = None


        self.states = states
        self.s0     = s0
        self.msgs   = msgs

        debugPrint(                                   \
            "\n\nstates = "      + str(self.states) + \
            "\n\ns0 = "          + str(self.s0)     + \
            "\n\ntransitions = " + str(self.transitions))

        self.machine = Machine(\
            model=self,        \
            states=self.states,\
            initial=self.s0)

        for (a, b, c) in self.transitions:
            
            debugPrint(str(a) + "," + str(b) + "," + str(c))
            
            if a in self.states and c in self.states:

                self.machine.add_transition(\
                    trigger=b, 
                    source=a, 
                    dest=c)
            else:
                debugPrint(                         \
                    "ERROR: (a, b, c) = (" + str(a) \
                    + ", " + str(b) + ", " + str(c) \
                    + ") in states = " + str(self.states), True)

    def toPromela(self, numPeers=2, tranFilter=lambda t : True):
        cleaned_transitions = [(a, B.replace("-", "_"), c) 
                               for (a, B, c) in self.transitions]

        cleaned_states = [s.replace("-", "_") for s in self.states]

        cleaned_s0 = self.s0.replace("-", "_")

        msgs = self.msgs

        if msgs == None:
            msgs = list(set([b for (a, B, c) in cleaned_transitions \
                             for b in B.split(';')                  \
                             if b.lower() != "timeout" ]))
            msgs = list(set([m[:-1] for m in msgs if m != 'ε']))

        body = "mtype = { "
        body += ", ".join(msgs) + " }\n"

        allowedTransitions = [t for t in cleaned_transitions if tranFilter(t)]

        if numPeers > 2:
            print("Currently toPromela() cannot handle numPeers > 2.")
            assert(False) # to throw an error

        elif numPeers == 1:
            body += "chan c = [1] of { mtype }\n"
            body += proctype(cleaned_states, allowedTransitions)

        elif numPeers == 2:
            body += "chan AtoN = [1] of { mtype }\n"
            body += "chan NtoA = [0] of { mtype }\n"
            body += "chan BtoN = [1] of { mtype }\n"
            body += "chan NtoB = [0] of { mtype }\n"

            body += network("AtoN", "NtoA", "BtoN", "NtoB", msgs)
            
            body += proctype(cleaned_states, 
                             allowedTransitions, 
                             "peerA", 
                             "NtoA", 
                             "AtoN",
                             cleaned_s0)

            body += proctype(cleaned_states, 
                             allowedTransitions, 
                             "peerB", 
                             "NtoB", 
                             "BtoN",
                             cleaned_s0)

        return body

    def compareTo(self, other,      \
        showit=False,               \
        printit=True,               \
        comm_transitions=None,      \
        user_call_transitions=None, \
        rst_transitions=None,       \
        transition_filter=lambda x : True):

        S1 , S2  = self.states     , other.states
        T1 , T2  = self.transitions, other.transitions

        # we will now print a graph s.t.: 
        # any node       in both self and other is BLUE
        blue_nodes        = set(S1).intersection(set(S2))
        # any transition in both self and other is BLUE
        blue_transitions  = set(T1).intersection(set(T2)) 
        # any node       in self but not other is RED
        red_nodes         = set(S1) - set(S2)
        # any transition in self but not other is RED
        red_transitions   = set(T1) - set(T2) 
        # any node       in other but not self is GREEN
        green_nodes       = set(S2) - set(S1)
        # any transition in other but not self is GREEN
        green_transitions = set(T2) - set(T1)
        
        if showit == True:
            showgraph(self,             \
                      other,            \
                      blue_nodes,       \
                      blue_transitions, \
                      red_nodes,        \
                      red_transitions,  \
                      green_nodes,      \
                      green_transitions)

        # PRINT SUMMARY
        if printit == True:

            printSummaryOfDifference(  \
                blue_nodes,            \
                blue_transitions,      \
                red_nodes,             \
                red_transitions,       \
                green_nodes,           \
                green_transitions,     \
                self.getLabelMap(),    \
                other.getLabelMap(),   \
                self.removedLines,     \
                comm_transitions,      \
                user_call_transitions, \
                rst_transitions,       \
                transition_filter)

    def writePromela(self, name, numPeers=2, tranFilter=lambda t : True):
        write_name = name + ".pml"
        with open(write_name, "w") as fw:
            to_write = self.toPromela(numPeers, tranFilter)
            fw.write(to_write)

    def writeImage(self, name):
        self.get_graph().draw(name + ".png", prog='dot')

    def save(self, writepng, writepromela, name, \
             numPeers=2,                         \
             tranFilter=lambda t : True):
        if writepng:
            self.writeImage(name)
        if writepromela:
            self.writePromela(name, numPeers, tranFilter)

    def getLabelMap(self):
    	return self.labelmap

def network(AtoN, NtoA, BtoN, NtoB, symbols):
    ret = "\nactive proctype network() {\n\tdo"
    for (inC, outC) in [(AtoN, NtoB), (BtoN, NtoA)]:
        for symbol in symbols:
            ret += "\n\t:: " + inC + " ? " + symbol \
                 + " -> \n\t\tif\n\t\t:: " \
                 + outC \
                 + " ! " \
                 + symbol \
                 + ";\n\t\tfi unless timeout;\n"
    ret += "\n\tod\n}\n"
    return ret


def proctype(states, transitions, name="translated", inC="c", outC="c", s0=None):

    if (s0 == None):
        print("Error: no initial state supplied to proctype().  "\
              + "Defaulting to states[0] = " + str(states[0]))

    body = "active proctype " + name + "(){\n"
    body += "\tgoto " + s0.upper() + ";\n"; # go to initial state s0
    # ORDER THE STATES FOR DETERMINISM!!!
    states = sorted(states)
    for state in states:
        Any = len([s for s in transitions if s[0] == state]) > 0
        body += state + ":\n\tif\n"
        for (start, label, end) in transitions:
            if start == state:
                body += "\t:: "
                sub_labels = [l for l in label.split(";") if l != 'ε']
                if len(sub_labels) == 0:
                    body += "goto " + end + ";\n"
                else:
                    for sub in sub_labels:
                        if sub[-1] == "?":
                            body += inC + " ? " + sub[:-1] + "; "
                        elif sub[-1] == "!":
                            body += outC + " ! " + sub[:-1] + "; "
                    body += "goto " + end + ";\n"
        if Any == False:
            body += "\t:: skip;\n"
        body += "\tfi\n"
    body += "}\n"
    return body
