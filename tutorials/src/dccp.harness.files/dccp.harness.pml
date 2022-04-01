mtype = { DCCP_REQUEST, 
          DCCP_RESPONSE, 
          DCCP_DATA, 
          DCCP_ACK, 
          DCCP_DATAACK, 
          DCCP_CLOSEREQ, 
          DCCP_CLOSE, 
          DCCP_RESET,
          DCCP_SYNC,
          DCCP_SYNCACK };

chan AtoN = [1] of { mtype };
chan NtoA = [0] of { mtype };
chan BtoN = [1] of { mtype };
chan NtoB = [0] of { mtype };

int state[2];
int before_state[2];

#define ClosedState    0
#define ListenState    1
#define RequestState   2
#define RespondState   3
#define PartOpenState  4
#define OpenState      5
#define CloseReqState  6
#define ClosingState   7
#define TimeWaitState  8

#define StableState    9
#define ChangingState  10
#define UnstableState  11
#define EndState       -1

#define leftClosed       (state[0] == ClosedState)
#define rightEstablished (state[1] == OpenState)

#define leftListen   (state[0] == ListenState)
#define leftTimeWait (state[0] == TimeWaitState)
#define leftRespond  (state[0] == RespondState)
#define leftLTR      (leftListen || leftTimeWait || leftRespond)
#define leftTR       (              leftTimeWait || leftRespond)

bit b = 0;

proctype DCCP(chan snd, rcv; int i) {
    bool I_am_active;
CLOSED:
    I_am_active = false;
    before_state[i] = state[i];
    state[i] = ClosedState;
    if
    :: goto LISTEN; /* passive open */
    :: snd ! DCCP_REQUEST;  /* active  open */ 
       goto REQUEST; 
    fi
LISTEN:
    before_state[i] = state[i];
    state[i] = ListenState;
    if
    :: rcv ? DCCP_REQUEST -> /* rcv request  */
       snd ! DCCP_RESPONSE; /* snd response */ 
       goto RESPOND;
    :: timeout -> goto CLOSED; // need to add this to the FSM & confirm
    fi
REQUEST:
    I_am_active = true;
    before_state[i] = state[i];
    state[i] = RequestState;
    if
    :: rcv ? DCCP_RESPONSE -> /* rcv response */
       snd ! DCCP_ACK;       /* snd Ack      */
       goto PARTOPEN;
    :: rcv ? DCCP_RESET -> goto CLOSED;
    :: rcv ? DCCP_SYNC -> snd ! DCCP_RESET; goto CLOSED;
    :: timeout -> goto CLOSED; // need to add this to the FSM & confirm
    fi
RESPOND:
    I_am_active = false;
    before_state[i] = state[i];
    state[i] = RespondState;
    /* rcv Ack/DataAck */
    do
    :: rcv ? DCCP_ACK     -> goto OPEN;
    :: rcv ? DCCP_DATAACK -> goto OPEN; 
    /* It MAY also leave the RESPOND state for CLOSED after a timeout of not less
     * than 4MSL (8 minutes); when doing so, it SHOULD send a DCCP-Reset
     * with Reset Code 2, "Aborted", to clean up state at the client. */
    :: timeout -> 
        if
        :: snd ! DCCP_RESET;
        :: skip;
        fi
        goto CLOSED;
    :: snd ! DCCP_DATA; // need to add this to the FSM & confirm
    od
PARTOPEN:
    before_state[i] = state[i];
    state[i] = PartOpenState;
    do
    /* rcv packet */
    :: rcv ? DCCP_DATA;    snd ! DCCP_ACK; goto OPEN;
    :: rcv ? DCCP_DATAACK; snd ! DCCP_ACK; goto OPEN;
    /* send packet */
    :: snd ! DCCP_DATAACK;
    /* timeout for reliability */
    :: timeout -> goto CLOSED;
    /* go to OPEN because they know about me -- but do so implicitly, because
     * it would be kind of inconvenient to process a message twice in Promela.
     */
    :: rcv ? DCCP_CLOSEREQ -> snd ! DCCP_CLOSE; goto CLOSING; // duplicate OPEN logic
    :: rcv ? DCCP_CLOSE    -> snd ! DCCP_RESET; goto CLOSED;  // duplicate OPEN logic
    :: rcv ? DCCP_ACK      -> goto OPEN;
    od
OPEN:
    before_state[i] = state[i];
    state[i] = OpenState;
    do
    /* send data */
    :: snd ! DCCP_DATA;
    :: snd ! DCCP_DATAACK;
    :: rcv ? DCCP_ACK;
    :: rcv ? DCCP_DATA;
    :: rcv ? DCCP_DATAACK;
    /* server active close */
    :: I_am_active == true -> 
       snd ! DCCP_CLOSEREQ; /* snd CloseReq */
       goto CLOSEREQ;
    :: rcv ? DCCP_CLOSE -> /* rcv Close */
       snd ! DCCP_RESET; /* snd Reset */
       goto CLOSED;
    /* active close */
    :: I_am_active == true ->
       snd ! DCCP_CLOSE; /* snd Close */ 
       goto CLOSING;
    :: rcv ? DCCP_CLOSEREQ -> /* rcv CloseReq */
       snd ! DCCP_CLOSE;    /* snd Close */ 
       goto CLOSING;
    /* simply shut down the connection silently */
    :: goto CLOSED; // need to add this to the FSM & confirm 
    od
CLOSEREQ:
    before_state[i] = state[i];
    state[i] = CloseReqState;
    rcv ? DCCP_CLOSE;  /* rcv Close */
    snd ! DCCP_RESET; /* snd Reset */
    goto CLOSED;
CLOSING:
    before_state[i] = state[i];
    state[i] = ClosingState;
    if
    :: rcv ? DCCP_RESET -> /* rcv Reset */
       goto TIMEWAIT;
    // Not in the spec
    :: timeout -> goto CLOSED;
    fi
TIMEWAIT:
    before_state[i] = state[i];
    state[i] = TimeWaitState;
    skip; /* 2MSL timer expires*/
    goto CLOSED;
}

init {
    state[0] = ClosedState;
    state[1] = ClosedState;
    before_state[0] = ClosedState;
    before_state[1] = ClosedState;
    run DCCP(AtoN, NtoA, 0);
    run DCCP(BtoN, NtoB, 1);
}

ltl all_phi {
    (eventually ( b == 1 ) ) implies (
    	(
    	    ! ( eventually 
    	        always (
    	            (state[0] == before_state[0]) &&
    	            (state[1] == before_state[1])
    	        )
    	    )
        )
        &&
        (
    		always (
            	!(state[0] == TimeWaitState && state[1] == TimeWaitState)
        	)
        )
        &&
        (
        	!(eventually always (state[0] == before_state[0]))
        )
        &&
        (
        	always (
    			!(
    				state[0] == CloseReqState &&
    				state[1] == CloseReqState
    			)
    		)	
        )
    )
}

