"""
100:	proc  3 (TCP:1) TEMPORARY.pml:79 (state 42)	[state[i] = 5]
		queue 3 (AtoN): 
		queue 1 (NtoA): [ACK]
		queue 4 (BtoN): [FIN]
		queue 2 (NtoB): [ACK]
		state[0] = 1
		state[1] = 5
		pids[0] = 2
		pids[1] = 3
		TCP(3):i = 1
		queue 2 (TCP(3):rcv): [ACK]
		queue 4 (TCP(3):snd): [FIN]
"""
import sys
from tabulate import tabulate


def is_send_event(event_text):
	return "BtoN!" in event_text or \
	       "AtoN!" in event_text or \
	       "NtoA!" in event_text or \
	       "NtoB!" in event_text

def is_rcv_event(event_text):
	return "BtoN?" in event_text or \
	       "AtoN?" in event_text or \
	       "NtoB?" in event_text or \
	       "NtoA?" in event_text

def is_state_assignment_zero(event_text):
	return "state[0] =" in event_text

def is_state_assignment_one(event_text):
	return "state[1] =" in event_text

def is_queue_status(event_text):
	return "queue " in event_text

def get_state_number(event_text):
	try:
		return event_text.split("state[")[1]\
		                 .split("=")[1]\
		                 .split("]")[0]\
		                 .strip()
	except:
		return "couldn't split {" + event_text + "}"

STATE_MAP = {
	"0":"CLOSED",
	"1":"LISTEN",
	"2":"SYN_SENT",
	"3":"SYN_RECEIVED",
	"4":"ESTABLISHED",
	"5":"FIN_WAIT_1",
	"6":"CLOSE_WAIT",
	"7":"FIN_WAIT_2",
	"8":"CLOSING",
	"9":"LAST_ACK",
	"10":"TIME_WAIT",
	"-1":"END"
}

def print_receive_event(event_text):
	pre = None
	post = None
	if "AtoN?" in event_text:
		pre = "Network received "
		post = " from Peer0 for Peer1"
	elif "BtoN?" in event_text:
		pre = "Network received "
		post = " from Peer1 for Peer0"
	elif "NtoA?" in event_text:
		pre = "Peer1 received "
		post = " via Network"
	elif "NtoB?" in event_text:
		pre = "Peer2 received "
		post = " via Network"
	msg_num = event_text.split("?")[1].strip()
	if "]" in msg_num:
		msg_num = msg_num.split("]")[0]
	return str(pre) + str(msg_num) + str(post)

def print_send_event(event_text):
	pre = None
	post = None
	if "AtoN!" in event_text:
		pre = "Peer0 sent "
		post = " to Peer1 over Network"
	elif "BtoN!" in event_text:
		pre = "Peer1 sent "
		post = " to Peer0 over Network"
	elif "NtoA!" in event_text:
		pre = "Network delivered "
		post = " to Peer0"
	elif "NtoB!" in event_text:
		pre = "Network delivered "
		post = " to Peer1"
	msg_num = event_text.split("!")[1].strip()
	if "]" in msg_num:
		msg_num = msg_num.split("]")[0]
	return str(pre) + str(msg_num) + str(post)

# https://stackoverflow.com/a/480227/1586231
def unq(seq):
    last = None
    newlst = []
    for item in seq:
    	if last == None:
    		newlst.append(item)
    	elif last != item:
    		newlst.append(item)
    	last = item
    return newlst

def main():

	p0, p1, = "NULL", "NULL"
	AtoN, BtoN, NtoA, NtoB = "[   ]", "[   ]", "[   ]", "[   ]"

	states = []

	for line in sys.stdin:
		stripped = line.strip()
		
		# [queue 4 (BtoN): [FIN]]
		if "(BtoN): " in stripped:
			BtoN = "[" + stripped.split("(BtoN): ")[1].split("[")[1].split("]")[0] + "]"

		elif "(AtoN): " in stripped:
			AtoN = "[" + stripped.split("(AtoN): ")[1].split("[")[1].split("]")[0] + "]"

		elif "(NtoA): " in stripped:
			NtoA = "[" + stripped.split("(NtoA): ")[1].split("[")[1].split("]")[0] + "]"

		elif "(NtoB): " in stripped:
			NtoB = "[" + stripped.split("(NtoB): ")[1].split("[")[1].split("]")[0] + "]"
		
		elif is_state_assignment_zero(line):
			p0 = STATE_MAP[get_state_number(stripped)]

		elif is_state_assignment_one(line):
			p1 = STATE_MAP[get_state_number(stripped)]
			states.append([p0, AtoN, NtoA, NtoB, BtoN, p1])

		elif "<<<<<START OF CYCLE>>>>>" in line:
			states.append(["...", "...", "...", "...", "...", "..."]) # indicating cycle start
			

	print(tabulate(unq(states), headers=["Peer0", "0-->N", "0<--N", "N-->1", "N<--1", "Peer1"]))

if __name__ == "__main__":
    # execute only if run as a script
    main()