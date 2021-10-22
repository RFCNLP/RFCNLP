"""
file       : analyzeOutput.py
author     : [redacted]
authored   : 23 July 2021
description: Creates results tables for paper, from captured CLI output.
"""
from glob     import glob
from tabulate import tabulate

from printUtils import deColor, makeGreen, makeRed, makeBlue, makeFail, makeHeader

def modelSupportTable(resultsFolder="RESULTS/", protocol="TCP", colored=True):
    
    NotModelsPattern = protocol                                  + \
                       ".pml"                                    + \
                       ("[93m" if colored else "")         + \
                       "⊭"                                       + \
                       ("[0m" if colored else "")          + \
                       "promela-models/" + \
                       protocol                                  + \
                       "/props/phi"
    
    ModelsPattern    = protocol                                  + \
                       ".pml"                                    + \
                       ("[92m" if colored else "")         + \
                       "⊨"                                       + \
                       ("[0m" if colored else "")          + \
                       "promela-models/" + \
                       protocol                                  + \
                       "/props/phi"
    
    model_logs = glob(resultsFolder + protocol.lower() + "*2promela.txt")

    model_to_yes_no = { t : { } for t in model_logs }

    all_props = set()

    for model_log in model_logs:
        
        with open(model_log, "r") as fr:
            
            for line in fr:
                
                if NotModelsPattern in line:

                    assert(not ModelsPattern in line)
                    prop = "phi" + line.split(NotModelsPattern)[1].strip()
                    model_to_yes_no[model_log][prop] = makeRed("False")
                    all_props.add(prop)

                elif ModelsPattern in line:

                    assert(not NotModelsPattern in line)
                    prop = "phi" + line.split(ModelsPattern)[1].strip()
                    model_to_yes_no[model_log][prop] = makeGreen("True")
                    all_props.add(prop)
    
    all_props = sorted(list(all_props))
    header    = [ "Model" ] + all_props
    rows      = []
    
    for model in model_to_yes_no:
        new_row = [ model ]
        for prop in all_props:
            if prop in model_to_yes_no[model]:
                new_row.append(model_to_yes_no[model][prop])
            else:
                assert(False) # this should not happen
        rows.append(new_row)

    print("\n" + tabulate(rows, headers=header) + "\n")


"""
------- TCP-(Gold, Linear, Bert) "Hard-Transition" Attacks with Recovery -------
"""
def attacksWithRecovery(
    resultsFolder="RESULTS/", 
    protocol="TCP",
    canonicalModel="promela-models/TCP/Canonical-Test/Canonical-TCP-test.pml",
    colored=True):
    
    pre_pattern = ("[94m" if colored else "") + "[ comparing to " + canonicalModel + " ]"
    
    model_logs  = glob(resultsFolder + protocol.lower() + "*2promela.txt")

    header = [ 
        "Model", 
        "Attack Number", 
        "From Prop", 
        "To Prop", 
        "Soft/Hard",
    ]
    
    rows   = [ ]
    
    doStep1 = False
    is_recovery_A = False

    cur_row = [ None, None, None, None, None ]
    
    for model_log in model_logs:
        with open(model_log, "r") as fr:
            for line in fr:
                
                if doStep1 == True:

                    is_recovery_A = "is an attack with recovery against" in line
                    is_soft       = "_soft_transitions.pml" in line

                    from_property = line.split(".props.phi")[1].split("-")[0]
                    from_property = "phi" + from_property + ".pml"

                    attack_number = line.split("/attacker_")[1].split("_")[0]
                    
                    if is_recovery_A == True:
                        assert(cur_row == [ None, 
                                            None, 
                                            None, 
                                            None, 
                                            None ])
                        cur_row[1]    = attack_number
                        cur_row[2]    = from_property
                        cur_row[4] = "Soft" if is_soft else "Hard"

                elif is_recovery_A == True:

                    assert(cur_row[0] == None and \
                           cur_row[1] != None and \
                           cur_row[2] != None and \
                           cur_row[3] == None and \
                           cur_row[4] != None)

                    cur_row[0] = model_log.split(resultsFolder)[1]\
                                          .split("2promela"   )[0]\
                                          .replace(protocol.lower(), 
                                                     protocol + "_")

                    if cur_row[0].split("_")[1].strip() == "":
                        cur_row[0] += "gold"

                    to_property   = line.split("phi"    )[1]\
                                        .split("."      )[0]
                    to_property   = "phi" + to_property + ".pml"
                    cur_row[3]    = to_property

                    rows.append([ deColor(r) for r in cur_row ])
                    
                    cur_row = [ None, None, None, None, None ]

                    is_recovery_A = False

                doStep1 = pre_pattern in line

    rows = sorted(rows)

    print(tabulate(rows, headers=header))

    fileNameToModelName = lambda r : r.split("_")[1]

    sub_models     = sorted(set([r[0].split("_")[1] for r in rows]))
    sub_props_to   = sorted(list(set([r[3] for r in rows])))
    sub_props_from = sorted(list(set([r[2] for r in rows])))

    for TransitionType in "Hard", "Soft":

        coloredTransitionType = makeBlue(TransitionType) \
                                if TransitionType == "Hard" else \
                                makeFail(TransitionType)

        print("\n\n" + protocol + "-(" + ", ".join(sub_models) + ") " + \
            coloredTransitionType + \
            " transition attacks with recovery")
        
        sum_header = [ "From \\ To " ] + sub_props_to

        sum_rows = [
            [ prop_from ] + [
                tuple([
                    len([r for r in rows if \
                         r[2] == prop_from      and \
                         r[3] == prop_to        and \
                         r[4] == TransitionType and \
                         fileNameToModelName(r[0]) == sub_model])

                    for sub_model in sub_models

                ])
                for prop_to in sub_props_to
            ] for prop_from in sub_props_from
        ]

        print(tabulate(sum_rows, headers=sum_header))


CANONICAL_TCP  = "promela-models/TCP/Canonical-Test/Canonical-TCP-test.pml"
CANONICAL_DCCP = "promela-models/DCCP/Canonical-Test-Client-Server/" + \
                 "canonical.DCCP.client.server.pml" 

modelSupportTable("RESULTS.july.27/", "TCP")  # table 1
modelSupportTable("RESULTS.july.27/", "DCCP") # table 2

attacksWithRecovery("RESULTS.july.27/", "TCP",  CANONICAL_TCP)
attacksWithRecovery("RESULTS.july.27/", "DCCP", CANONICAL_DCCP)

