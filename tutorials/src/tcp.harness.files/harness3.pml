mtype = { SYN, FIN, ACK, ABORT, CLOSE, RST, OPEN }

chan AtoN = [1] of { mtype };
chan NtoA = [0] of { mtype };
chan BtoN = [1] of { mtype };
chan NtoB = [0] of { mtype };

int state[2];
int pids[2];

#define ClosedState    0
#define ListenState    1
#define SynSentState   2
#define SynRecState    3
#define EstState       4
#define FinW1State     5
#define CloseWaitState 6
#define FinW2State     7
#define ClosingState   8
#define LastAckState   9
#define TimeWaitState  10
#define EndState       -1

#define leftConnecting (state[0] == ListenState && state[1] == SynSentState)
#define leftEstablished (state[0] == EstState)
#define rightEstablished (state[1] == EstState)
#define leftClosed (state[0] == ClosedState)

bit b = 0;

proctype TCP(chan snd, rcv; int i) {
	pids[i] = _pid;
CLOSED:
	state[i] = ClosedState;
	if
	/* Passive open */
	:: goto LISTEN;
	/* Active open */
	:: snd ! SYN; goto SYN_SENT;
	/* Terminate */
	:: goto end;
	fi
LISTEN:
	state[i] = ListenState;
	if
	:: rcv ? SYN -> snd ! SYN; 
	                snd ! ACK; goto SYN_RECEIVED;
	/* Simultaneous LISTEN */
	:: timeout -> goto CLOSED; 
	fi
SYN_SENT:
	state[i] = SynSentState;
	if
	:: rcv ? SYN;
		if
		/* Standard behavior */
		:: rcv ? ACK -> snd ! ACK; goto ESTABLISHED;
		/* Simultaneous open */
		:: snd ! ACK; goto SYN_RECEIVED;
		fi
	:: rcv ? ACK; rcv ? SYN -> snd ! ACK; goto ESTABLISHED;
	/* Timeout */
	:: timeout -> goto CLOSED;
	fi
SYN_RECEIVED:
	state[i] = SynRecState;
	rcv ? ACK; goto ESTABLISHED;
ESTABLISHED:
	state[i] = EstState;
	if
	/* Close - initiator sequence */
	:: snd ! FIN; goto FIN_WAIT_1;
	/* Close - responder sequence */
	:: rcv ? FIN -> snd ! ACK; goto CLOSE_WAIT;
	fi
FIN_WAIT_1:
	state[i] = FinW1State;
	if
	/* Simultaneous close */
	:: rcv ? FIN -> snd ! ACK; goto CLOSING;
	/* Standard close */
	:: rcv ? ACK -> goto FIN_WAIT_2;
	fi
CLOSE_WAIT:
	state[i] = CloseWaitState;
	snd ! FIN; goto LAST_ACK;
FIN_WAIT_2:
	state[i] = FinW2State;
	rcv ? FIN -> snd ! ACK; goto TIME_WAIT;
CLOSING:
	state[i] = ClosingState;
	rcv ? ACK -> goto TIME_WAIT;
LAST_ACK:
	state[i] = LastAckState;
	rcv ? ACK -> goto CLOSED;
TIME_WAIT:
	state[i] = TimeWaitState;
	goto CLOSED;
end:
	state[i] = EndState;
}

init {
	state[0] = ClosedState;
	state[1] = ClosedState;
	run TCP(AtoN, NtoA, 0);
	run TCP(BtoN, NtoB, 1);
}

ltl all_phi {
	(eventually ( b == 1 ) ) implies (
		!(eventually (((always (state[0] == SynSentState))   ||
		               (always (state[0] == SynRecState))    ||
		               (always (state[0] == EstState))       ||
		               (always (state[0] == FinW1State))     ||
		               (always (state[0] == CloseWaitState)) ||
		               (always (state[0] == FinW2State))     ||
		               (always (state[0] == ClosingState))   ||
		               (always (state[0] == LastAckState))   ||
		               (always (state[0] == TimeWaitState)))
		               &&
		              ((always (state[1] == SynSentState))   ||
		               (always (state[1] == SynRecState))    ||
		               (always (state[1] == EstState))       ||
		               (always (state[1] == FinW1State))     ||
		               (always (state[1] == CloseWaitState)) ||
		               (always (state[1] == FinW2State))     ||
		               (always (state[1] == ClosingState))   ||
		               (always (state[1] == LastAckState))   ||
		               (always (state[1] == TimeWaitState)))))
	)
}