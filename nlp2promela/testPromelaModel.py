# coding: utf-8
'''
name       : testPromelaModel.py
author     : [redacted]
authored   : 17 October 2020
description: tests the output TCP model
'''

from korg.Characterize import *
from korg.Korg         import body

from printUtils        import makeGreen, makeRed
from testConstants     import TCP
from TCPtestConstants  import canonical_TCP_model, \
                              canonical_TCP_IO,    \
                              TCP_props_path,      \
                              TCP_props,           \
                              TCP_AP_prefix
from DCCPtestConstants import canonical_DCCP_model, \
                              canonical_DCCP_IO,    \
                              DCCP_props_path,      \
                              DCCP_props,           \
                              DCCP_AP_prefix

import os
import glob
import shutil

def bigCleanUp(cur_model_name):
    for file in glob.glob("*.trail") + \
                glob.glob("*tmp*")   + \
                glob.glob("pan")     + \
                glob.glob("*.pml")   + \
                glob.glob("._n_i_p_s_"):
        if file != cur_model_name + ".pml" and \
           file != cur_model_name + "_CORRECT.pml":
            os.remove(file)

# Function to check if P |= phi
def models2(P, phi, cur_model_name):
    # CLEAN
    bigCleanUp(cur_model_name)
    # RUN
    tmpname = str(abs(hash(P))) + "." + str(abs(hash(phi))) + ".pml"
    while os.path.isfile(tmpname):
        tmpname = str(abs(hash(tmpname))) + ".pml"
    with open(tmpname, "w") as fw:
        with open(P, "r") as fr:
            for line in fr:
                fw.write(line)
        with open(phi, "r") as fr:
            for line in fr:
                fw.write(line)
    return check(tmpname)

# Function to evaluate what properties TCP model |='s.
def what_properties_does_TCP_model_support(P):
    return [models2(P, phi, "TCP") for phi in TCP_props]

def choose_a_tmpname(P, midfix):
    tmpname = P.replace(".pml", "") + "." + midfix + ".pml"
    while os.path.isfile(tmpname):
        tmpname = str(abs(hash(tmpname))) + ".pml"
    return tmpname

def add_AP_to(P, AP, tmpname, atomic_propositions_prefix):
    i = 0
    tmpdir = "TEMPORARY_added_ap_" + str(abs(hash(tmpname))) + "/"
    os.mkdir(tmpdir)
    tmpname = tmpdir + tmpname
    with open(tmpname, "w") as fw:
        fw.write(atomic_propositions_prefix)
        with open(P, "r") as fr:
            for line in fr:
                if len([key for key in AP if key + ":" == line.strip()]) > 0:
                    for key in AP:
                        if key + ":" == line.strip():
                            (value, i) = AP[key]
                            if "before_state" in atomic_propositions_prefix:
                                bef = "before_state[" + str(i) + "] = state[" + str(i) + "];"
                                aft = "state[" + str(i) + "] = " + value + ";"
                                fw.write(key + ":\n\t" + bef + "\n\t" + aft + "\n")
                            else:
                                fw.write(key + ":\n\tstate[" + str(i) + "] = " + value + ";\n")
                            AP[key] = (value, i + 1)
                            break
                else:
                    fw.write(line)
    return tmpname

# Function to add atomic propositions to DCCP model.
def add_atomic_propositions_to_DCCP_model(P):
    tmpname = choose_a_tmpname(P, "AP")
    AP = {
        'CLOSED'   : ('ClosedState'  , 0), 
        'LISTEN'   : ('ListenState'  , 0), 
        'REQUEST'  : ('RequestState' , 0), 
        'RESPOND'  : ('RespondState' , 0), 
        'PARTOPEN' : ('PartOpenState', 0), 
        'OPEN'     : ('OpenState'    , 0),
        'CLOSEREQ' : ('CloseReqState', 0),
        'CLOSING'  : ('ClosingState' , 0),
        'TIMEWAIT' : ('TimeWaitState', 0), 
        'STABLE'   : ('StableState'  , 0), 
        'CHANGING' : ('ChangingState', 0), 
        'UNSTABLE' : ('UnstableState', 0)
    }
    return add_AP_to(P, AP, tmpname, DCCP_AP_prefix)

# Function to add atomic propositions to TCP model.
def add_atomic_propositions_to_TCP_model(P):
    tmpname = choose_a_tmpname(P, "AP")
    AP = {
        'CLOSED'       : ('ClosedState'   , 0),
        'LISTEN'       : ('ListenState'   , 0),
        'SYN_SENT'     : ('SynSentState'  , 0),
        'SYN_RECEIVED' : ('SynRecState'   , 0),
        'ESTABLISHED'  : ('EstState'      , 0),
        'FIN_WAIT_1'   : ('FinW1State'    , 0),
        'FIN_WAIT_2'   : ('FinW2State'    , 0),
        'CLOSE_WAIT'   : ('CloseWaitState', 0),
        'CLOSING'      : ('ClosingState'  , 0),
        'LAST_ACK'     : ('LastAckState'  , 0),
        'TIME_WAIT'    : ('TimeWaitState' , 0),
        'end'          : ('EndState'      , 0)
    }
    return add_AP_to(P, AP, tmpname, TCP_AP_prefix)

def parse_2_peer_model(the_model):
    network_code   = ""
    remainder_code = ""
    with open(the_model, "r") as fr:
        in_network = False
        for line in fr:
            if (line.strip() == "}" and in_network == True):
                in_network = False
                network_code += line
            elif (line.strip() == "active proctype network() {"):
                in_network = True
                network_code += line
            elif (in_network == False):
                remainder_code += line
            elif (in_network == True):
                network_code += line
    net_name = str(abs(hash(network_code))) + ".pml"
    rem_name = str(abs(hash(remainder_code))) + ".pml"

    dir_name_prefix = "TEMPORARY-net-rem-"
    dir_name = dir_name_prefix + str(abs(hash(net_name + rem_name)))
    while os.path.isdir(dir_name):
        dir_name = dir_name_prefix + str(abs(hash(dir_name)))
    os.mkdir(dir_name)

    with open(dir_name + "/" + net_name, "w") as fw:
        fw.write(network_code)

    print("WROTE TO " + dir_name + "/" + net_name)

    with open(dir_name + "/" + rem_name, "w") as fw:
        fw.write(remainder_code)

    print("WROTE TO " + dir_name + "/" + rem_name)

    return (dir_name + "/" + net_name, dir_name + "/" + rem_name)


def auto_evaluate_some_pml(some_pml, l, Phi, canonical_IO, canonical_model, cur_model_name):
    print("\n+++++++++++++++++++++++++ SUPPORTED PROPERTIES +++++++++++++++++++++++++\n")
    labeled = l(some_pml)
    (network, remainder) = parse_2_peer_model(labeled)

    the_name = some_pml.replace(".pml", "")
    if "/" in the_name:
        the_name = the_name.split("/")[-1]
    
    attack_these = set()
    for phi in Phi:
        support = models2(labeled, phi, cur_model_name)
        if support == True:
            print(some_pml + makeGreen("⊨") + phi)
            attack_these.add(phi)
        elif support == False:
            print(some_pml + makeRed("⊭") + phi)
        else:
            print("ERROR!!! support = " + str(support))

    for prop in sorted(list(attack_these)):
        bigCleanUp(cur_model_name)
        
        print("============ TRY TO ATTACK " + prop + "============")
        print("remainder = " + remainder       )
        print(     "prop = " + prop            )
        print(  "network = " + network         )
        print(       "IO = " + canonical_IO    )
        
        dirname = "attack-"                                       \
                 + prop    .replace("/", ".").replace(".pml", "-") \
                 + some_pml.replace("/", ".").replace(".pml", "-")

        while (os.path.isdir("out/" + dirname)):
            dirname = str(abs(hash(dirname))) + "." + dirname

        body(remainder, 
             prop, 
             network, 
             canonical_IO, 
             max_attacks=100,
             with_recovery=True, 
             name=dirname, 
             characterize=False,
             comparisons=[canonical_model])
    
    bigCleanUp(cur_model_name)

def auto_evaluate_DCCP_pml(DCCP_pml):
    auto_evaluate_some_pml(DCCP_pml,                              \
                           add_atomic_propositions_to_DCCP_model, \
                           DCCP_props,                            \
                           canonical_DCCP_IO,                     \
                           canonical_DCCP_model,                  \
                           "DCCP")

# Function to do automatic evaluation
def auto_evaluate_TCP_pml(TCP_pml):
    auto_evaluate_some_pml(TCP_pml,                              \
                           add_atomic_propositions_to_TCP_model, \
                           TCP_props,                            \
                           canonical_TCP_IO,                     \
                           canonical_TCP_model,                  \
                           "TCP")