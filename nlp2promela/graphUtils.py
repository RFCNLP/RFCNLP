'''
name       : graphUtils.py
author     : [redacted]
authored   : 21 August 2020
description: provides graph utilities to fsmUtils
'''

import networkx  as nx
# import gmatch4py as gm

from tabulate import tabulate

import textwrap

from printUtils import debugPrint, printTransition, makeGreen, makeRed

"""
Technically this is an estimate, but I think it's close enough
for development purposes.  If we need a number in the final paper,
we should use the canonical nx.graph_edit_distance(dg1, dg2).

INPUTS:

  > dg1 - nx.DiGraph number one
  > dg2 - nx.DiGraph number two

OUTPUTS:

  > approximate edit_distance(dg1, dg2) as an integer string

"""
def graphDistance(dg1, dg2):
    # https://networkx.github.io/documentation/
    # stable/_modules/networkx/algorithms/similarity.html
    
    # ged = gm.GraphEditDistance(1,1,1,1)
    # mat = ged.compare([dg1, dg2], None)

    # estimate_1 = mat[0][1]
    # estimate_2 = mat[1][0]

    # return str(int((int(estimate_1) + int(estimate_2)) / 2))
    return "I don't know."

"""
INPUTS:

  > edges - some list of (source, label, dest)

OUTPUTS:

  > the list of (source, label, dest) in the
    format (source, dest, { 'label' : label })

Useful for wrangling NetworkX.
"""
edged = lambda edges : [(a, c, {'label' : b}) for (a, b, c) in edges]

"""
INPUTS:

  > blue_nodes        - nodes found in both dg1 and dg2
  > blue_transitions  - transitions found in both dg1 and dg2
  > red_nodes         - nodes found in dg1 but not in dg2
  > red_transitions   - transitions found in dg1 but not in dg2
  > green_nodes       - nodes found in dg2 but not in dg1
  > green_transitions - transitions found in dg2 but not in dg1

OUTPUTS:

  > Nothing, but has side-effect of printing the approximate edit
    distance between the two graphs (dg1 and dg2).

TODO: 

  See if we have already constructed these graphs elsewhere and
  can more easily pass them in without re-constructing them.
"""
def printGraphDistance(blue_nodes,       \
                       blue_transitions, \
                       red_nodes,        \
                       red_transitions,  \
                       green_nodes,      \
                       green_transitions):
    
    dg1 = nx.DiGraph()
    dg2 = nx.DiGraph()

    dg1.add_nodes_from(blue_nodes.union(red_nodes))
    dg2.add_nodes_from(blue_nodes.union(green_nodes))

    dg1.add_edges_from(edged(blue_transitions.union(red_transitions)))
    dg2.add_edges_from(edged(blue_transitions.union(green_transitions)))

    dist = graphDistance(dg1, dg2)
    
    print(dist)

def adjustedNumber(number, missing):
    # The integer number line is like:
    # [][][][][][][][][][][][][][][][][][][][][][]
    # but our number line is like
    # [][][]<>[][][][]<><>[][][][]<>[][][][]<><><>
    # where the <> are the missing numbers.
    # So, given the index in the line Z - missing,
    # what is the index in the original line Z?
    cur_z     = 0
    cur_z_mod = 0
    while cur_z_mod < number:
        cur_z += 1
        if not cur_z in missing:
            cur_z_mod += 1
    return cur_z - 1

def adjustLineNumbers(numbers, removedLines):
    return [adjustedNumber(n, removedLines) \
            for n in numbers]

"""
INPUTS:

  > blue_nodes        - nodes found in both dg1 and dg2
  > blue_transitions  - transitions found in both dg1 and dg2
  > red_nodes         - nodes found in dg1 but not in dg2
  > red_transitions   - transitions found in dg1 but not in dg2
  > green_nodes       - nodes found in dg2 but not in dg1
  > green_transitions - transitions found in dg2 but not in dg1
  > lm1               - labelmap of dg1
  > lm2               - labelmap of dg2

OUTPUTS:

  > Nothing, but has side-effect of printing a summary message
    describing and diagnosing the differences between dg1 and dg2.
"""
def printSummaryOfDifference(blue_nodes,                 \
                             blue_transitions,           \
                             red_nodes,                  \
                             red_transitions,            \
                             green_nodes,                \
                             green_transitions,          \
                             lm1,                        \
                             lm2,                        \
                             removedLines=None,          \
                             comm_transitions=None,      \
                             user_call_transitions=None, \
                             rst_transitions=None,       \
                             transition_filter=lambda x : True):

    isTCP = (not None in [comm_transitions,      \
                          user_call_transitions, \
                          rst_transitions])

    num_correct_comm_transitions      = 0
    num_missing_comm_transitions      = 0
    num_correct_user_call_transitions = 0
    num_missing_user_call_transitions = 0
    num_correct_rst_transitions       = 0
    num_missing_rst_transitions       = 0

    print("Graph distance = ")

    printGraphDistance(       \
            blue_nodes,       \
            blue_transitions, \
            red_nodes,        \
            red_transitions,  \
            green_nodes,      \
            green_transitions)

    printstates(blue_nodes , "CORRECT")
    printstates(red_nodes  , "WRONG"  )
    printstates(green_nodes, "MISSING")

    k = len(blue_transitions)
    print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~ " 
          + str(k) 
          + " CORRECT TRANSITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    if len([t for t in blue_transitions if not t in lm1]) == 0:
        toprintblue = [[], [], [], []] if isTCP else []
        
        colsblue = ["Source", "Label", "Destination", "Line #s"]
        
        for transition in sorted(blue_transitions):
            source, label, destination = transition
            originallines = lm1[(source, label, destination)]
            lines = str(originallines)
            if removedLines != None:
                lines += " ---> " 
                lines += str(adjustLineNumbers(originallines, removedLines))
            if isTCP:
                split_transition = (source, label.split(";"), destination)
                if split_transition in comm_transitions:
                    toprintblue[0].append([source, label, destination, lines])
                    num_correct_comm_transitions += 1
                elif split_transition in user_call_transitions:
                    toprintblue[1].append([source, label, destination, lines])
                    num_correct_user_call_transitions += 1
                elif split_transition in rst_transitions:
                    toprintblue[2].append([source, label, destination, lines])
                    num_correct_rst_transitions += 1
                else:
                    toprintblue[3].append([source, label, destination, lines])
            else:
                toprintblue.append([source, label, destination, lines])

        if isTCP:

            print("\n\t\t" 
                  + str(len(toprintblue[0])) 
                  + " Correct Communication Transitions")

            print(tabulate(toprintblue[0], colsblue, tablefmt="fancy_grid"))
            
            print("\n\t\t" 
                  + str(len(toprintblue[1])) 
                  + " Correct User Call Transitions")

            print(tabulate(toprintblue[1], colsblue, tablefmt="fancy_grid"))
            
            print("\n\t\t" 
                  + str(len(toprintblue[2])) 
                  + " Correct Reset Transitions")

            print(tabulate(toprintblue[2], colsblue, tablefmt="fancy_grid"))

            unknown_len = len(toprintblue[3])
            if (unknown_len > 0):

                print("\n\t\t" + str(unknown_len) + " Correct Transitions of Unknown Category")
                print(tabulate(toprintblue[3], colsblue, tablefmt="fancy_grid"))
        
        else:

            print(tabulate(toprintblue, colsblue, tablefmt="fancy_grid"))
    else:
        print(                                                            \
            tabulate(                                                     \
                sorted(                                                   \
                  [list(transition) for transition in blue_transitions]), \
                ["Source", "Label", "Destination"],                       \
                tablefmt="fancy_grid"))
    
    # Wrong transitions ...

    rows = []

    good_transitions = blue_transitions.union(green_transitions)

    def transition_filter_str(T):
        (a, l, c) = T
        passes_filter = transition_filter((a, l, c))
        if passes_filter:
            return makeGreen("Yes")
        else:
            return makeRed("No")

    for (a, l, c) in red_transitions:
        diagnosis = diagt((a, l, c),             \
                          good_transitions,      \
                          comm_transitions,      \
                          user_call_transitions, \
                          rst_transitions)

        if lm1 != None and (a, l, c) in lm1:
            num = -1
            lines = lm1[(a, l, c)]
            if removedLines != None:
                originalnums = adjustLineNumbers(lines, removedLines)
                lines = str(lines) + " ---> " + str(originalnums)
            rows.append([a, l, c, lines, diagnosis, transition_filter_str((a, l, c))])
        else:
            rows.append([a, l, c, ['?'], diagnosis, transition_filter_str((a, l, c))])

    cols = ["Source", "Label", "Destination", "Line #s", "Diagnosis", "Passes Filter?"]

    bad_filt = len([r for r in rows if "Yes" in r[-1]])
    bad_all = len(rows)

    print("\n\n\t\t" 
          + str(bad_all) 
          + " WRONG TRANSITIONS, of which " 
          + str(bad_filt) 
          + " pass the filter\n")
    print(tabulate(sorted(rows), cols, tablefmt="fancy_grid"))

    k = len(green_transitions)
    print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~ " 
          + str(k) 
          + " MISSING TRANSITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

    missing_transitions = [[], [], [], []] if isTCP else []
    for transition in sorted(green_transitions):
        if isTCP:
            (source, label, destination) = transition
            split_transition = (source, label.split(";"), destination)
            if split_transition in comm_transitions:
                missing_transitions[0].append([source, label, destination])
                num_missing_comm_transitions += 1
            elif split_transition in user_call_transitions:
                missing_transitions[1].append([source, label, destination])
                num_missing_user_call_transitions += 1
            elif split_transition in rst_transitions:
                missing_transitions[2].append([source, label, destination])
                num_missing_rst_transitions += 1
            else:
                missing_transitions[3].append([source, label, destination])
        else:
            missing_transitions.append([source, label, destination])
    
    missing_transitions_head = ["Source", "Label", "Destination"]
    
    if isTCP:
        print("\n\t\t" 
                  + str(len(missing_transitions[0])) 
                  + " Missing Communication Transitions")

        print(tabulate(missing_transitions[0], 
                       missing_transitions_head, 
                       tablefmt="fancy_grid"))
        
        print("\n\t\t" 
              + str(len(missing_transitions[1])) 
              + " Missing User Call Transitions")

        print(tabulate(missing_transitions[1], 
                       missing_transitions_head, 
                       tablefmt="fancy_grid"))
        
        print("\n\t\t" 
              + str(len(missing_transitions[2])) 
              + " Missing Reset Transitions")

        print(tabulate(missing_transitions[2], 
                       missing_transitions_head, 
                       tablefmt="fancy_grid"))

        unknown_len = len(missing_transitions[3])
        if (unknown_len > 0):

            print("\n\t\t" + str(unknown_len) + " Missing Transitions of Unknown Category")
            print(tabulate(missing_transitions[3], 
                           missing_transitions_head, 
                           tablefmt="fancy_grid"))
    else:
        print(                                                             \
            tabulate(missing_transitions,      \
                     missing_transitions_head, \
                     tablefmt="fancy_grid"))

    if isTCP:
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~ SUMMARY STATISTICS ~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        
        class1 = (num_correct_comm_transitions, \
                  num_missing_comm_transitions, \
                  len(comm_transitions),        \
                  "Communication")
        
        class2 = (num_correct_user_call_transitions, \
                  num_missing_user_call_transitions, \
                  len(user_call_transitions),        \
                  "User Calls")
        
        class3 = (num_correct_rst_transitions, \
                  num_missing_rst_transitions, \
                  len(rst_transitions),        \
                  "Resets")
        
        for classI in [class1, class2, class3]:
            right, wrong, exp, name = classI
            print("\nWe expect " + str(exp) + " " + name + " transitions.")
            print("\n\tOf those, we find "      \
                  + str(right)                \
                  + " but are still missing " \
                  + str(wrong)                \
                  + ".")



"""
INPUTS:

  > S - a list

OUTPUTS:

  > S if S is not empty else '∅'

PURPOSE:

  > Basically just a pretty-printer for lists.
"""
emptyifempty = lambda S : '∅' if len(S) == 0 else S

"""
INPUTS:

  > s - a list of states
  > n - some prefix message

OUTPUTS:

  > Nothing, but side-effect of printing a nice message 
    to communicate n with s.
"""
def printstates(s, n): 
    slst = emptyifempty(", ".join(s))
    if slst == "∅":
        print("\n\t" + n + " STATES: ∅")
    else:
        print("\n\t" + n + " STATES:")
        print("\n\t\t" + "\n\t\t".join(textwrap.wrap(slst, 80)))
"""
INPUTS:

  > self              - henceforth called dg1, a nx.DiGraph
  > other             - henceforth called dg2, a nx.DiGraph
  > blue_nodes        - nodes found in both dg1 and dg2
  > blue_transitions  - transitions found in both dg1 and dg2
  > red_nodes         - nodes found in dg1 but not in dg2
  > red_transitions   - transitions found in dg1 but not in dg2
  > green_nodes       - nodes found in dg2 but not in dg1
  > green_transitions - transitions found in dg2 but not in dg1

OUTPUTS:

  > nothing, but has side-effect of showing a graph comparing 
    dg1 and dg2.

NOTE:

  > Not yet done.
"""
def showgraph(self,             \
              other,            \
              blue_nodes,       \
              blue_transitions, \
              red_nodes,        \
              red_transitions,  \
              green_nodes,      \
              green_transitions):
    
    # make the graph
    G = nx.DiGraph()

    node_colors = []

    def _add_node(node, color):
        # is it self.s0?
        label = node
        if node == self.s0:
            label += " = self.s0"
        if node == other.s0:
            label += " = other.s0"
        # add the node
        G.add_node(node, label=label)
        node_colors.append(color)
    
    for node in blue_nodes:
        _add_node(node, 'blue')
    for node in red_nodes:
        _add_node(node, 'red')
    for node in green_nodes:
        _add_node(node, 'green')

    tran_colors = []

    def _add_transition(transition, color):
        G.add_edge(transition[0],            \
                   transition[2],            \
                   label=str(transition[1]))
        tran_colors.append(color)

    for tran in blue_transitions:
        _add_transition(tran, 'blue')
    for tran in red_transitions:
        _add_transition(tran, 'red')
    for tran in green_transitions:
        _add_transition(tran, 'green')

    # done
    nx.draw(G,                               \
            with_labels=True,                \
            node_color=node_colors,          \
            edge_color=tran_colors,          \
            connectionstyle='arc3, rad = 0.1')

    # TODO: figure out how to add the transition labels, which is non-
    #       trivial as we could have multiple different transitions w/
    #       same start, end but different labels ...

    plt.show()


def variety_of_t(a, b, c, comm, user, rst):
    bb = b.split(";")
    ret = []
    if (a, bb, c) in comm:
        ret.append("(Comm)")
    if (a, bb, c) in user:
        ret.append("(User)")
    if (a, bb, c) in rst:
        ret.append("(RST)")
    return ret

def varieties_of_T(T, comm, user, rst):
    varieties = []
    for (a, b, c) in T:
        append_varieties = variety_of_t(a, b, c, comm, user, rst)
        varieties += append_varieties
    return "[" + ",".join(list(set(varieties))) + "]"

"""
INPUTS:

  > t - a transition (source, label, dest)
  > T - a set of transitions from which t *should* be drawn

OUTPUTS:

  > a diagnosis explaining how to transform t into its pseudo-closest
    relative in T, heuristically.
"""
def diagt(t, T, comm=[], user=[], rst=[]):
    (a, b, c) = t 

    if (a, b, c) in T:
        return "CORRECT AS IS - see: " \
             + variety_of_t(a, b, c, comm, user, rst)
    
    swapped = [(x, y, z) for (x, y, z) in T if (x, y, z) == (c, b, a)]
    if len(swapped) > 0:
        return "SWAP START & END STATES - see: " \
             + varieties_of_T(swapped, comm, user, rst)
    
    wrongargtrans = [(x, y, z) for (x, y, z) in T if (x, z) == (a, c)]
    wrongargs     = [y for (x, y, z) in wrongargtrans]

    if len(wrongargs) > 0:
        return "SWAP ARG W/ SOME l ∈ "     \
             + str(wrongargs) + " - see: " \
             + varieties_of_T(wrongargtrans, comm, user, rst)
    
    wrongstarttrans = [(x, y, z) for (x, y, z) in T if (y, z) == (b, c)]
    wrongstart      = [x for (x, y, z) in wrongstarttrans]
    if len(wrongstart) > 0:
        return "SWAP START W/ SOME x ∈ " \
             + str(wrongstart) \
             + " - see: "      \
             + varieties_of_T(wrongstarttrans, comm, user, rst)
    
    wrongendtrans = [(x, y, z) for (x, y, z) in T if (x, y) == (a, b)]
    wrongend      = [z for (x, y, z) in wrongendtrans]
    if len(wrongend) > 0:
        return "SWAP END W/ SOME x ∈ " \
             + str(wrongend)           \
             + " - see: "              \
             + varieties_of_T(wrongendtrans, comm, user, rst)
    
    
    return "I DON'T KNOW ..."

"""
INPUTS:

  > t  - a transition (source, label, dest)
  > T  - a set of transitions from which t *should* be drawn
  > lm - a line map t -> lines(t)

OUTPUTS:

  > A pretty-printed diagnosis of the transition.

"""
def wrongformatter(t, T, lm=None, sz=55):
    label = printTransition(t)
    delta = sz - len(label)
    ans   = "?"
    if delta > 0:
        ans = label + delta * " " + "// " + diagt(t, T)
    else:
        ans = label + " // " + label
    if lm != None and t in lm:
        attribution = ""
        rel_lines = lm[t]
        if len(rel_lines) == 1:
            attribution = "check line: " \
                        + str(rel_lines[0])
        elif len(rel_lines) > 1:
            attribution = "check lines: "   \
                        + ", ".join([str(l) \
                                     for l  \
                                     in rel_lines])
        ans += "\n\t   " + sz * " " + attribution
    return ans