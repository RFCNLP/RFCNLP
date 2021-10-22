'''
name       : xmlUtils.py
author     : [redacted]
authored   : 9 June 2020
description: provides xml utils for nlp2promela
'''
from printUtils  import debugPrint
from stringUtils import bestGuess, cleanUp
from constants   import tags

from lxml import etree as ET

def hasParents(descendent, xml):
    parent_map = dict((c, p) for p in xml.getiterator() for c in p)
    return descendent in parent_map and parent_map[descendent] != descendent

def parentState(descendent, xml, filt=lambda x : True):
    parent_map = dict((c, p) for p in xml.getiterator() for c in p)
    if not descendent in parent_map:
        return None
    tmp = parent_map[descendent]
    if filt(tmp) == True:
        return tmp
    if tmp == None:
        return None
    return parentState(tmp, xml, filt)

def triggerEvents(trigger):
    return [e for e in trigger.iter('ref_event')]

# Returns reason, x
# where x = True  -> reason is of the form ( "a;b;c;..." )
# and   x = False -> reason is of the form ( "a" or "b" or "c" or ... )
def findReason(t, xml, id2reason, outer_control=None):

    all_possible_events = set()

    for v in id2reason.values():
        if v != "timeout":
            all_possible_events.add(v)

    # map for send/receive annotation
    omap = { "send" : "!", "receive" : "?", "compute" : "" }
    # find the actions to parse
    events = []
    saw_this_transition = False; or_logic = False
    other_than_receive = False; other_than_send = False
    for child in xml:
        
        if child == t:
            saw_this_transition = True
        
        if child.tag != "action":
           
            if child.text          != None  and \
               saw_this_transition == False and \
               child.tag == "trigger":

                trigger_text = str(child.text).lower()
                if "else" in trigger_text or "otherwise" in trigger_text:
                    
                    if (len(events) > 0):
                        
                        events_without_punctuation = [e.replace("!", "")\
                                                       .replace("?", "")\
                                                        for e in events]
                        
                        complement_reasons \
                            = list(all_possible_events \
                                   - set(events_without_punctuation))

                        complement_kinds = ["?"]
                        final_complement_reasons = list(set([
                            reason + kind for reason in complement_reasons
                                for kind in complement_kinds
                            if reason != "timeout"
                        ]))
                        return final_complement_reasons, False, or_logic

            elif child.text != None:
                trigger_text = str(child.text).lower()
                
                if "other than" in trigger_text and "receive" in trigger_text:
                    other_than_receive = True
                
                elif "other than" in trigger_text and "send" in trigger_text:
                    other_than_send = True

            for e in child.iter("ref_event"):
                
                try:
                    reason = id2reason[e.get("id")]

                    e_type = e.get("type")
                    argument = None
                    
                    # We consider "timeout" a special symbol with 
                    # known meaning across all protocols
                    if reason == "timeout":
                        argument = ""
                    
                    # When it is not a timeout, and we don't know the
                    # type, we assume it is a send, if it is not in an
                    # <action> block.
                    # elif e_type == "None":
                    #     argument = "!"

                    # Otherwise, it is not a timeout, and we do know the
                    # type (ie the argument), so we can just set it.
                    else:
                        argument = omap[e_type]
                    
                    if reason + argument not in events:
                        events.append(reason + argument)

                except Exception as ex:
                    pass
                    
        elif child.tag == "action" and child.get("type") != None:
            try:
                argument = omap[child.get("type")]
                for sub_child in child:
                    if sub_child.tag == "arg":
                        for sub_sub_child in sub_child.iter("ref_event"):
                            reason = id2reason[sub_sub_child.get("id")]
                            # Add a heuristic to not add duplicates
                            if reason + argument not in events:
                                events.append(reason + argument)
            except Exception as ex:
                pass

    if other_than_receive and len(events) > 0:
        
        complement_reasons = list(all_possible_events - set(events))
        complement_reasons = [r + "?" for r in complement_reasons]
        events = complement_reasons
        or_logic = True
    
    elif other_than_send and len(events) > 0:
        
        complement_reasons = list(all_possible_events - set(events))
        complement_reasons = [r + "!" for r in complement_reasons]
        events = complement_reasons
        or_logic = True

    if len(events) > 0:
        return events, True, or_logic

    return ['ε'], True, or_logic

def findExplicitToState(t, id2state):
    to_state = None
    find_target = t.find('arg_target')
    find_source_target = t.find('arg_source_target')
    
    if find_target is not None:
        for s in t.iter('arg_target'):
            for child in s.iter('ref_state'):
                chid = child.get('id')
                to_state = id2state[chid]
    
    elif find_source_target is not None:
        for s in t.iter('arg_source_target'):
            for child in s.iter('ref_state'):
                chid = child.get('id')
                to_state = id2state[chid]
    
    return to_state

def findExplicitFromState(t, id2state):
    from_state = None
    find_source = t.find('arg_source')
    find_source_target = t.find('arg_source_target')
    
    if find_source is not None:
        for ta in t.iter('arg_source'):
            for child in ta.iter('ref_state'):
                chid = child.get('id')
                from_state = id2state[chid]
    
    elif find_source_target is not None:
        for ta in t.iter('arg_source_target'):
            for child in ta.iter('ref_state'):
                chid = child.get('id')
                from_state = id2state[chid]
    
    return from_state

def findExplicitIntermediateStates(t, id2state):
    intermediate = []
    try:
        for i in t.iter('arg_intermediate'):
            for child in i.iter('ref_state'):
                chid = child.get('id')
                intermediate.append(id2state[chid])
    except:
        return intermediate
    return intermediate

def findInnerState(t, id2state):
    to_state = None
    try:
        for child in t.iter('ref_state'):
            chid = child.get('id')
            to_state = id2state[chid]
    except:
        return to_state
    return to_state

def findOuterStates(t, ctl, id2state):
    if ET.tostring(t) == ET.tostring(ctl):
        return []
    options, cur_option = [], []
    for child in ctl:
        if child.tag == 'ref_state' or child.tag == 'def_state':
            cid = child.get('id')
            state = id2state[cid]
            cur_option.append(state)
        elif len(cur_option) > 0:
            options.append(cur_option)
            cur_option = []
        elif child.tag != 'transition':
            sub_options = findOuterStates(t, child, id2state)
            if sub_options != None and len(sub_options) > 0:
                options.append(sub_options)
            cur_option = []
    if len(cur_option) > 0:
        options.append(cur_option)
    options = [o for o in options if o != None]
    if len(options) == 0:
        return []
    bst, val = None, -1
    for option in options:
        oval = len(options)
        if oval > val:
            val = oval
            bst = option
    return bst

def nearestControl(base, xml):
    control = parentState(base, xml)
    while control != None and not (control.tag in { "control", "p" }):
        control = parentState(control, xml)
    return control

# When calling, initialize map_state to:
# id2state = { a.get("id") : a.text for a in xml.iter('def_state') }

def splitArguments(argument):
    ret = []; current = []
    prev_arg = argument[0]
    current.append(prev_arg)
    # Prefer ? ! than single ? or !
    # Prefer ? or ! than consecutive ? ? or ! !
    for curr_arg in argument[1:]:
        if (prev_arg.endswith('!') and curr_arg.endswith('?')) or\
           (prev_arg.endswith('!') and curr_arg.endswith('!')) or\
           (prev_arg.endswith('?') and curr_arg.endswith('?')):
            ret.append(current)
            current = []
        current.append(curr_arg)
        prev_arg = curr_arg
    ret.append(current)
    return ret

def parseTransition(t, xml, id2reason, id2state, trygoup=True, numup=6):
    # find explicit states - in some cases, both are explicit
    to_state = findExplicitToState(t, id2state)
    from_state = findExplicitFromState(t, id2state)
    intermediate_states = findExplicitIntermediateStates(t, id2state)
    intermediate_states = [s.upper() for s in intermediate_states]

    outer_states = []
    # find upper bound
    control = nearestControl(t, xml)
    # find argument
    argument, subsequent, or_logic \
        = findReason(t, control, id2reason, nearestControl(control, xml))

    # This is the case we had in TCP where annotations were not explicit 
    # (we assume it is always a target)
    if to_state is None and from_state is None:
        # find to_state
        to_state = findInnerState(t, id2state)
        # find from_states
        outer_states = findOuterStates(t, control, id2state)
    # This case is similar to the above but we have an explicit -target- or 
    # -source- annotation inside <transition>
    # In DCCP, there are a few annotations where the single reference inside is 
    # a source
    elif to_state is None or from_state is None:
        # find outside states
        outer_states = findOuterStates(t, control, id2state)
    #print("to_state", to_state, "from_state", from_state)

    # Try going up just one layer.
    i = 1
    while (argument == ['ε'] or len(outer_states) == 0) \
         and trygoup \
         and i < numup \
         and subsequent == True:

        parent_control = nearestControl(control, xml)

        if parent_control != control and \
           parent_control != None and \
           parent_control.tag == "control":

            if argument == ['ε']:
                argument, subsequent, or_logic = findReason(t,              \
                                                            parent_control, \
                                                            id2reason)

            if len(outer_states) == 0 and (to_state   is None or \
                                           from_state is None):
                
                outer_states = findOuterStates(control,        \
                                               parent_control, \
                                               id2state)

        else:
            break

        i += 1
        control = parent_control

    # upper case
    outer_states = [s.upper() for s in outer_states]

    if to_state is None and from_state is not None and len(outer_states) > 0:
        '''
        if len(outer_states) > 1:
            Currently ignoring this case 
            (it doesn't happen in DCCP annotations)
            pass
        '''
        to_state = outer_states[0].upper()
        from_states = [from_state.upper()]
    elif to_state is not None and from_state is None:
        to_state = to_state.upper()
        from_states = outer_states
    elif to_state is not None and from_state is not None:
        to_state = to_state.upper()
        from_states = [from_state.upper()]
    elif to_state is None and from_state is None:
        from_states = outer_states
    else:
        from_states = []

    # Check for unexplained context
    unexplained_context = findPrependEventsByContext(control, xml, id2reason)
    if (len(unexplained_context) > 0):
        transition_string = str(from_states) \
                          + " ---"           \
                          + str(argument)    \
                          + "---> "          \
                          + str(to_state)
        
        print("\nPossibly" + transition_string + " should be " \
              + str(from_states)                               \
              + " ---"                                         \
              + str(unexplained_context + argument)            \
              + "---> "                                        \
              + str(to_state)                                  \
              + " ? Check on this.")

    if len(intermediate_states) > 0:

        if subsequent == False:
            print("Error !! We have not thought about, let alone engineered "
                + "for, the scenario where you have an ~otherwise~ and a "
                + "~multi-part~ all in the same <control>...<transition>..."
                + "</transition>...</control>... Hence, in this case, the "
                + "results will almost certainly be flawed."
                + " Please refer to parseTransition(...) in xmlUtils.py.")

        argument_splits = splitArguments(argument)
        inter_state = intermediate_states[0]
        prev_inter_state = inter_state

        if len(argument_splits) == (len(intermediate_states) + 1):
            # The guessed split of arguments matches the number of hops
            ret = [(from_states, argument_splits[0], inter_state)]
            
            for inter_state, arg in zip(intermediate_states[1:], \
                                        argument_splits[1:]):
                
                ret.append(([prev_inter_state], arg, inter_state))
                prev_inter_state = inter_state
            
            ret.append(([inter_state], argument_splits[-1], to_state))
        
        else:
            # Otherwise, resort to vanilla solution
            ret = [(from_states, argument, inter_state)]
            for inter_state in intermediate_states[1:]:
                ret.append(([prev_inter_state], argument, inter_state))
                prev_inter_state = inter_state
            ret.append(([inter_state], argument, to_state))

        return ret
    
    elif subsequent == True and or_logic == False:
    
        return [(from_states, argument, to_state)]
    
    elif subsequent == True and or_logic == True:

        ret = [(from_states, [a], to_state) for a in argument]
        return ret

    elif subsequent == False:

        control = nearestControl(t, xml)
        outer_control = nearestControl(control, xml)
        for con in [ control, outer_control ]:
            for ot in con.iter():
                if ot == t:
                    break
                if ot.tag == "trigger":
                    candidate_from_state = findExplicitFromState(ot, id2state)
                    if candidate_from_state != None:
                        from_states = [candidate_from_state]
                if from_states != []:
                    break

        ret = [(from_states, [a], to_state) for a in argument]
        print("\n\nret = ", ret)
        return ret

def guessInitial(xml):
    defStates = [(s, s.text) for s in xml.iter('def_state')]
    guesses   = { a : 0 for (a, _) in defStates }
    for (s, t) in defStates:
        if "initial" in t.lower():
            guesses[t] += 1
        paren = parentState(s, xml, lambda x : (x.tag in tags))
        if paren      != None and \
           paren.text != None and \
           "begin" in paren.text.lower():
            guesses[t] += 0.5
    i, v = None, 0
    for guess, value in guesses.items():
        if value > v:
            i = guess
            v = value
    if v == 0:
        return defStates[0][1]
    return v

# Returns a list of ref_events and def_events not in
# a control block.
def unexplainedEventsOf(e, id2reason):
    omap = { "send" : "!", "receive" : "?" }
    if e.tag in {'ref_event', 'def_event' }:
        if e.get("type") != "None":
            return [id2reason[e.get("id")] + omap[e.get("type")]]
        else:
            return []
    events = []
    for c in e:
        if c.tag != 'control':
            events += unexplainedEventsOf(c, id2reason)
    return events

# Makes sure no transition descendent exists not wrapped in
# a control.  Does not consider the context of e!!!
def containsUncontrolledTransition(e):
    for c in e:
        if c.tag == 'transition':
            return True
        if c.tag != 'control' and containsUncontrolledTransition(c):
            return True
    return False

"""
Consider the following pseudo-scenario.

More generally:

<control> <-----------------------------------------╮
  ...                                               |
  <ref_event ...>...</re_event>     <--- e0         |
  ...                                               |
  <control> <-------------------------------╮       |
      ...                                   |       | C_out
      <transition>...</transition>  <--- t0 | C_in  |
      ...                                   |       |
  </control> <------------------------------╯       |
  ...                                               |
</control> <----------------------------------------╯

When we parse something like this, we might find everything we
need to reconstruct a correct-looking transition inside C_in.
But what if really, we should also include e0 from C_out? 
How can we tell if this is the situation we find ourselves in?

Here is my solution (which is a little ad-hoc ...).
"""
def findPrependEventsByContext(C_in, root, id2reason):
    # Is there a control one-level up? If not, give up.
    C_out = parentState(C_in, root)
    if C_out == None or C_out == C_in or C_out.tag != 'control':
        return []
    # If we have made it this far, we found another control layer.
    # Are there any transitions in this control layer, prior to C_in,
    # that are not in their own control layers?
    unexplained_events = []
    for c in C_out:
        if c == C_in:
            break
        if containsUncontrolledTransition(c):
            return []
        unexplained_events += unexplainedEventsOf(c, id2reason)
    return unexplained_events
    

