'''
name       : nlp2xml.py
author     : [redacted]
authored   : 5 May 2020
updated    : 9 June 2020
description: translates annotated txt -> xml Python object
usage      : python nlp2promela.py rfcs-annotated-tidied/TCP.xml
'''
import sys
import os

from stringUtils      import cleanFile
from fsmUtils         import FSM
from xmlUtils         import *
from printUtils       import printTransition, printHeuristicRemoval
from testConstants    import *
from testPromelaModel import *

# INPUT:
#     we expect only a single argument, namely the path to the rfc .txt file
# OUTPUT:
#     a .png file in the directory from which the script was run, with the same
#     name as the input file, modulo extension.  This image represents
#     graphically the FSM of the protocol, or at least, my best guess.
def main(writepromela=True, writepng=True):
    rfc = sys.argv[1]
    cleanedFile, removedLines = cleanFile(rfc)
    xml = ET.parse(cleanedFile)
    
    S  = list(set([a.text.upper() for a in xml.iter('def_state')]))
    s0 = guessInitial(xml)

    states  = [cleanUp(s) for s in S]
    initial = cleanUp(s0)

    promela_convenience = lambda x : "timeout" if x == "timeout" else x.upper()

    id2state  = { a.get("id") : a.text for a in xml.iter('def_state') }
    id2reason = { a.get("id") : promela_convenience(a.text) for a in xml.iter('def_event') }

    parsed = lambda t : parseTransition(t, xml, id2reason, id2state)

    oktran = lambda A, b, c : len(A) > 0 and not None in [ b, c ]

    transitions = []

    for t in xml.iter('transition'):

        l = t.sourceline
        # Allow the case where a single tag retrieves more than one transition
        # (e.g. go from S through I to T)
        parsed_transitions = parsed(t)
        for (A, B, c) in parsed_transitions:
            if oktran(A, B, c):
                for a in A:
                    source = a.replace("-", "_")
                    dest   = c.replace("-", "_")
                    if not (a == c and (B == 'ε' or B == ['ε'])):
                        transition = (source, B, dest)
                        transtuple = (transition, l)
                        transitions.append(transtuple)

    # Heuristic: If we have S0 -- x? y! --> S1, 
    #                       S0 -- x?    --> S1, and 
    #                       S0 -- y!    --> S1
    # Then we can delete the latter 2 which are clearly noise caused by mis-parsing
    # of mentions of the first (presumably elsewhere in the text.)

    call_and_response_component_artifacts = [
        ((source, B, dest), l) for ((source, B, dest), l) in transitions
        if ((len(B) == 1) and
            any(
            [1 for ((s, BB, d), _) in transitions if 
                (set(B).issubset(set(BB)) and
                any([b for b in BB if b[-1] == "!"]) and
                any([b for b in BB if b[-1] == "?"]) and
                s == source                          and
                d == dest)
            ]
        ))
    ]

    # Heuristic: If we have x --(u)--> z and x --(ε)--> z, 
    #            then delete x --(ε)--> z because it's likely
    #            noise caused by the correct transition x --(u)--> z.
    redundant_epsilon_artifacts = [
        ((source, B, dest), l) for ((source, B, dest), l) in transitions
        if (B == ['ε']) and
            any(
                [1 for ((s, BB, d), _) in transitions if 
                (BB != ['ε'] and s == source and d == dest)])
    ]

    name = os.path.splitext(os.path.basename(rfc))[0]

    test_transitions = DCCP_transitions() if "DCCP" in name else \
                       TCP_communication_transitions() + \
                       TCP_user_call_transitions()     + \
                       TCP_rst_transitions() if "TCP" in name else []

    
    for c in call_and_response_component_artifacts:
        printHeuristicRemoval(c, "call-and-response", test_transitions)
        transitions.remove(c)

    for c in redundant_epsilon_artifacts:
        printHeuristicRemoval(c, "redundant-epsilon", test_transitions)
        transitions.remove(c)

    msgs = TCP_msgs()  if "TCP"  in name else \
           DCCP_msgs() if "DCCP" in name else \
           None

    result = FSM(states,                    \
                 initial,                   \
                 transitions,               \
                 labeled=True,              \
                 removedLines=removedLines, \
                 msgs=msgs)

    if "TCP" in name:
        result.save(writepng, writepromela, name, 2, TCPtranFilterHonest)
        testTCP(result, TCPtranFilterHonest)
        auto_evaluate_TCP_pml(name + ".pml")
        if writepromela:
            result.save(False, True, name,              2, TCPtranFilterHonest)
            TCP().save( False, True, name + "_CORRECT", 2, TCPtranFilterHonest)
    
    elif "DCCP" in name:
        result.save(writepng, writepromela, name, 2)
        testDCCP(result)
        auto_evaluate_DCCP_pml(name + ".pml")
        if writepromela:
            result.save(False, True, name, 2)
            DCCP().save(False, True, name + "_CORRECT", 2)
    
    else:
        result.save(writepng, writepromela, name)

if __name__ == "__main__":
    main()
