'''
name       : testConstants.py
author     : [redacted]
authored   : 9 August 2020
description: provides canonical models for comparison in unit-testing
'''
from fsmUtils import *
from fsmUtils import FSM

def DCCP_states():
	return ['CLOSED', 'LISTEN', 'REQUEST', 'RESPOND', 'PARTOPEN', 'OPEN', \
	        'CLOSEREQ', 'CLOSING', 'TIMEWAIT', 'STABLE', 'CHANGING', 'UNSTABLE']

def TCP_states():
	return ['CLOSED', 'SYN_SENT', 'LISTEN', 'CLOSING', 'TIME_WAIT', 'LAST_ACK', \
		   'ESTABLISHED', 'SYN_RECEIVED', 'FIN_WAIT_1', 'FIN_WAIT_2', \
		   'CLOSE_WAIT']

def DCCP_communication_labels():
	return { 'DCCP-REQUEST', 'DCCP-RESPONSE', 'DCCP-DATA', 'DCCP-ACK', \
	         'DCCP-DATAACK', 'DCCP-CLOSEREQ', 'DCCP-CLOSE', 'DCCP-RESET', \
	         'DCCP-SYNC', 'DCCP-SYNCACK' }

def TCP_communication_labels():
	return { 'SYN', 'ACK', 'FIN', 'SYN_ACK' }

# INCOMPLETE - missing data communication transitions in OPEN and
# PARTOPEN states.  Needs to be updated once I resolve bugs in Promela model.
def DCCP_transitions():
	return [# Transitions from the CLOSED state. Notice that the "OPEN"-
	        # label is just a convenience for an epsilon transition (OPEN
	        # is an entirely internal command, not a message).  I only
	        # include it so that [redacted]'s code works.
			('CLOSED'       , ['ε'                  ], 'LISTEN'      ), \
	        ('CLOSED'       , ['DCCP-REQUEST!'      ], 'REQUEST'     ), \
	        # Transitions from the LISTEN state.
	        ('LISTEN'       , ['DCCP-REQUEST?',     \
	        	               'DCCP-RESPONSE!'     ], 'RESPOND'     ), \
	        ('LISTEN'       , ['timeout'            ], 'CLOSED'      ), \
	        # Transitions from the REQUEST state.
	        ('REQUEST'      , ['DCCP-RESPONSE?',    \
	        	               'DCCP-ACK!'          ], 'PARTOPEN'    ), \
	        ('REQUEST'      , ['DCCP-RESET?'        ], 'CLOSED'      ), \
	        ('REQUEST'      , ['DCCP-SYNC?',        \
	                           'DCCP-RESET!'        ], 'CLOSED'      ), \
	        # Transitions from the RESPOND state.
	        ('RESPOND'      , ['DCCP-ACK?'          ], 'OPEN'        ), \
	        ('RESPOND'      , ['DCCP-DATAACK?'      ], 'OPEN'        ), \
	        ('RESPOND'      , ['timeout',           \
	        	               'DCCP-RESET!'        ], 'CLOSED'      ), \
	        # Transitions from the PARTOPEN state.
	        ('PARTOPEN'     , ['DCCP-ACK!'          ], 'PARTOPEN'    ), \
	        ('PARTOPEN'     , ['DCCP-DATA?',        \
	                           'DCCP-ACK!'          ], 'OPEN'        ), \
	        ('PARTOPEN'     , ['DCCP-DATAACK?',     \
	        	               'DCCP-ACK!'          ], 'OPEN'        ), \
	        ('PARTOPEN'     , ['DCCP-DATAACK!'      ], 'PARTOPEN'    ), \
	        ('PARTOPEN'     , ['timeout'            ], 'CLOSED'      ), \
	        ('PARTOPEN'     , ['DCCP-CLOSEREQ?',    \
	        	               'DCCP-CLOSE!'        ], 'CLOSING'     ), \
	        ('PARTOPEN'     , ['DCCP-CLOSE?',       \
	        	               'DCCP-RESET!'        ], 'CLOSED'      ), \
	        # If you receive anything other than a response or a sync, you
	        # go to OPEN.
	        ('PARTOPEN'     , ['DCCP-REQUEST?'      ], 'OPEN'        ), \
	        ('PARTOPEN'     , ['DCCP-DATA?'         ], 'OPEN'        ), \
	        ('PARTOPEN'     , ['DCCP-ACK?'          ], 'OPEN'        ), \
	        ('PARTOPEN'     , ['DCCP-DATAACK?'      ], 'OPEN'        ), \
	        # Transitions from the OPEN state.
	        ('OPEN'         , ['DCCP-DATA!'         ], 'OPEN'        ), \
	        ('OPEN'         , ['DCCP-DATAACK!'      ], 'OPEN'        ), \
	        ('OPEN'         , ['DCCP-ACK?'          ], 'OPEN'        ), \
	        ('OPEN'         , ['DCCP-DATA?'         ], 'OPEN'        ), \
	        ('OPEN'         , ['DCCP-DATAACK?'      ], 'OPEN'        ), \
	        ('OPEN'         , ['DCCP-CLOSEREQ!'     ], 'CLOSEREQ'    ), \
	        ('OPEN'         , ['DCCP-CLOSE?',       \
	        	               'DCCP-RESET!'        ], 'CLOSED'      ), \
	        ('OPEN'         , ['DCCP-CLOSE!'        ], 'CLOSING'     ), \
	        ('OPEN'         , ['DCCP-CLOSEREQ?',    \
	        	               'DCCP-CLOSE!'        ], 'CLOSING'     ), \
	        # Transitions from the CLOSEREQ state.
	       	('CLOSEREQ'     , ['DCCP-CLOSE?',       \
	                           'DCCP-RESET!'        ], 'CLOSED'      ), \
	       	('CLOSEREQ'     , ['DCCP-CLOSEREQ?'     ], 'CLOSING'     ), \
	        ('CLOSING'      , ['DCCP-RESET?'        ], 'TIMEWAIT'    ), \
	        ('TIMEWAIT'     , ['timeout'            ], 'CLOSED'      )]

def DCCP_msgs():
	return [ "DCCP_REQUEST", "DCCP_RESPONSE", "DCCP_DATA", "DCCP_ACK", \
	         "DCCP_DATAACK", "DCCP_CLOSEREQ", "DCCP_CLOSE", "DCCP_RESET", \
	         "DCCP_SYNC", "DCCP_SYNCACK" ]

def TCP_communication_transitions():
	return [('CLOSED'      , ['SYN!'                ], 'SYN_SENT'    ), \
	        ('SYN_SENT'    , ['ACK?'    , 'ACK!'    ], 'ESTABLISHED' ), \
	        ('SYN_SENT'    , ['SYN?'    , 'ACK!'    ], 'SYN_RECEIVED'), \
	        ('LISTEN'      , ['SYN?'    , 'ACK!'    ], 'SYN_RECEIVED'), \
	        ('LISTEN'      , ['SYN!'                ], 'SYN_SENT'    ), \
	        # Note that the CLOSING--ACK?--TIME_WAIT transition is 
		    # collapsed into the TIME_WAIT--CLOSED transition in our 
		    # SafeComp paper, because it adds an additional state without
		    # impacting the logic at the level we were evaluating.
		    # However, it is part of the FSM, so here we are explicitly
		    # including it.
		    ('CLOSING'     , ['ACK?'                ], 'TIME_WAIT'   ), \
		    ('CLOSING'     , ['FIN?'    , 'ACK!'    ], 'CLOSING'     ), \
		    ('TIME_WAIT'   , ['FIN?'    , 'ACK!'    ], 'TIME_WAIT'   ), \
		    ('LAST_ACK'    , ['ACK?'                ], 'CLOSED'      ), \
		    ('LAST_ACK'    , ['FIN?'    , 'ACK!'    ], 'LAST_ACK'    ), \
		    ('ESTABLISHED' , ['FIN?'    , 'ACK!'    ], 'CLOSE_WAIT'  ), \
		    ('ESTABLISHED' , ['FIN!'                ], 'FIN_WAIT_1'  ), \
		    ('SYN_RECEIVED', ['ACK?'                ], 'ESTABLISHED' ), \
		    ('FIN_WAIT_1'  , ['FIN?'    , 'ACK!'    ], 'CLOSING'     ), \
		    ('CLOSE_WAIT'  , ['FIN!'                ], 'LAST_ACK'    ), \
		    ('CLOSE_WAIT'  , ['FIN!'                ], 'CLOSING'     ), \
		    ('CLOSE_WAIT'  , ['FIN?'    , 'ACK!'    ], 'CLOSE_WAIT'  ), \
		    ('FIN_WAIT_1'  , ['ACK?'                ], 'FIN_WAIT_2'  ), \
		    ('FIN_WAIT_2'  , ['FIN?'    , 'ACK!'    ], 'TIME_WAIT'   ), \
		    ('SYN_RECEIVED', ['FIN?'    , 'ACK!'    ], 'CLOSE_WAIT'  )]

def TCP_msgs():
	return ["SYN", "ACK", "FIN"]

def TCP_user_call_transitions():
	return [# User OPEN call
		   ('CLOSED'      , ['OPEN?'               ], 'LISTEN'      ), \
		   # CLOSE CALLS
		   ('LISTEN'      , ['CLOSE?'              ], 'CLOSED'      ), \
		   ('SYN_SENT'    , ['CLOSE?'              ], 'CLOSED'      ), \
		   ('SYN_RECEIVED', ['CLOSE?'  , 'FIN!'    ], 'FIN_WAIT_1'  ), \
		   ('ESTABLISHED' , ['CLOSE?'  , 'FIN!'    ], 'FIN_WAIT_1'  ), \
		   ('CLOSE_WAIT'  , ['CLOSE?'  , 'FIN!'    ], 'CLOSING'     ), \
		   # ABORT CALLS
		   ('LISTEN'      , ['ABORT?'              ], 'CLOSED'      ), \
		   ('SYN_SENT'    , ['ABORT?'              ], 'CLOSED'      ), \
		   ('SYN_RECEIVED', ['ABORT?'  , 'RST!'    ], 'CLOSED'      ), \
		   ('ESTABLISHED' , ['ABORT?'  , 'RST!'    ], 'CLOSED'      ), \
		   ('FIN_WAIT_1'  , ['ABORT?'  , 'RST!'    ], 'CLOSED'      ), \
		   ('FIN_WAIT_2'  , ['ABORT?'  , 'RST!'    ], 'CLOSED'      ), \
		   ('CLOSE_WAIT'  , ['ABORT?'  , 'RST!'    ], 'CLOSED'      ), \
		   ('CLOSING'     , ['ABORT?'              ], 'CLOSED'      ), \
		   ('LAST_ACK'    , ['ABORT?'              ], 'CLOSED'      ), \
		   ('TIME_WAIT'   , ['ABORT?'              ], 'CLOSED'      )]

def TCP_rst_transitions():
	return [('LISTEN'      , ['RST?'                ], 'LISTEN'      ), \
		    ('LISTEN'      , ['RST?'                ], 'CLOSED'      ), \
		    ('SYN_SENT'    , ['ACK?'    , 'RST?'    ], 'CLOSED'      ), \
		    ('SYN_RECEIVED', ['RST?'                ], 'LISTEN'      ), \
		    ('SYN_RECEIVED', ['RST?'                ], 'CLOSED'      ), \
		    ('ESTABLISHED' , ['RST?'                ], 'CLOSED'      ), \
		    ('FIN_WAIT_1'  , ['RST?'                ], 'CLOSED'      ), \
		    ('FIN_WAIT_2'  , ['RST?'                ], 'CLOSED'      ), \
		    ('CLOSE_WAIT'  , ['RST?'                ], 'CLOSED'      ), \
		    ('CLOSING'     , ['RST?'                ], 'CLOSED'      ), \
		    ('LAST_ACK'    , ['RST?'                ], 'CLOSED'      ), \
		    ('TIME_WAIT'   , ['RST?'                ], 'CLOSED'      )]

def TCP():
	# see page 14 of https://arxiv.org/pdf/2004.01220.pdf
	# adapted with minor changes: no End state, no intermediary states ...
	states = TCP_states()

	transitions = TCP_communication_transitions() + \
	              TCP_user_call_transitions()     + \
	              TCP_rst_transitions()

	s0 = 'CLOSED'

	return FSM(states, 
		       s0, 
		       transitions, 
		       False,
		       None, 
		       TCP_msgs())

def DCCP():
	return FSM(DCCP_states(), 
		       'CLOSED', 
		       DCCP_transitions(),
		       False,
		       None,
		       DCCP_msgs())

def testProtocol(result, comparison, name, comm_transitions,     \
	                                       user_call_transitions,\
	                                       rst_transitions,      \
	             tranFilter = lambda x : True):
	comparison.get_graph().draw(name + "_CORRECT.png", prog='dot')
	with open(name + "_CORRECT.pml", "w") as fw:
		fw.write(comparison.toPromela())
	result.compareTo(\
		comparison,
		showit=False,
		printit=True,
		comm_transitions=comm_transitions,
		user_call_transitions=user_call_transitions,
		rst_transitions=rst_transitions,
		transition_filter=tranFilter)

def testDCCP(result):
	testProtocol(result, DCCP(), "DCCP", DCCP_transitions(), [], [])


def testTCP(result, tranFilter=lambda x : True):
	testProtocol(result, TCP(), "TCP", TCP_communication_transitions(), \
		                               TCP_user_call_transitions(),     \
		                               TCP_rst_transitions(),           \
		         tranFilter)

def testDCCP(result):
	comparison = DCCP()
	comparison.get_graph().draw("DCCP_CORRECT.png", prog='dot')
	with open("DCCP_CORRECT.pml", "w") as fw:
		fw.write(comparison.toPromela())
	result.compareTo(                        \
		comparison,                          \
		showit=False,                        \
		printit=True,                        \
		comm_transitions=DCCP_transitions(), \
		user_call_transitions=[],            \
		rst_transitions=[])

# USED in TCP debugging
def TCPtranFilter(t, T=TCP_communication_transitions()):
    (s, l, d) = t
    return (s, l.split(";"), d) in T

def DCCPtranFilter(t):
	return TCPtranFilter(t, T=DCCP_transitions())

# NOT CHEATING compared to TCPtranFilter
def TCPtranFilterHonest(t, L=TCP_communication_labels()):
	(s, l, d) = t 
	events = l.split(";")
	labels = [e.replace("!", "").replace("?", "") for e in events]
	comm_labels = L
	for l in labels:
		if l not in comm_labels:
			return False
	return True
