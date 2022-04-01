mtype = { SYN, ACK, FIN }
chan AtoN = [1] of { mtype }
chan NtoA = [0] of { mtype }
chan BtoN = [1] of { mtype }
chan NtoB = [0] of { mtype }

active proctype network() {
	do
	:: AtoN ? SYN -> 
		if
		:: NtoB ! SYN;
		fi unless timeout;

	:: AtoN ? ACK -> 
		if
		:: NtoB ! ACK;
		fi unless timeout;

	:: AtoN ? FIN -> 
		if
		:: NtoB ! FIN;
		fi unless timeout;

	:: BtoN ? SYN -> 
		if
		:: NtoA ! SYN;
		fi unless timeout;

	:: BtoN ? ACK -> 
		if
		:: NtoA ! ACK;
		fi unless timeout;

	:: BtoN ? FIN -> 
		if
		:: NtoA ! FIN;
		fi unless timeout;

	od
}
active proctype peerA(){
	goto LISTEN;
CLOSED:
	if
	:: AtoN ! SYN; goto LISTEN;
	:: AtoN ! SYN; goto SYN_SENT;
	fi
CLOSE_WAIT:
	if
	:: AtoN ! FIN; goto CLOSING;
	:: NtoA ? FIN; AtoN ! ACK; NtoA ? ACK; goto CLOSE_WAIT;
	fi
CLOSING:
	if
	:: NtoA ? ACK; goto TIME_WAIT;
	:: NtoA ? FIN; AtoN ! ACK; NtoA ? ACK; goto CLOSING;
	fi
ESTABLISHED:
	if
	:: AtoN ! FIN; goto FIN_WAIT_1;
	:: NtoA ? FIN; AtoN ! ACK; NtoA ? ACK; goto CLOSE_WAIT;
	fi
FIN_WAIT_1:
	if
	:: NtoA ? ACK; goto FIN_WAIT_2;
	:: NtoA ? ACK; goto TIME_WAIT;
	fi
FIN_WAIT_2:
	if
	:: NtoA ? FIN; AtoN ! ACK; NtoA ? ACK; goto TIME_WAIT;
	fi
LAST_ACK:
	if
	:: NtoA ? ACK; goto CLOSED;
	:: NtoA ? FIN; AtoN ! ACK; NtoA ? ACK; goto LAST_ACK;
	fi
LISTEN:
	if
	:: AtoN ! SYN; goto SYN_SENT;
	fi
SYN_RECEIVED:
	if
	:: NtoA ? ACK; goto ESTABLISHED;
	:: NtoA ? FIN; AtoN ! ACK; NtoA ? ACK; goto CLOSE_WAIT;
	fi
SYN_SENT:
	if
	:: AtoN ! ACK; goto SYN_RECEIVED;
	fi
TIME_WAIT:
	if
	:: NtoA ? FIN; AtoN ! ACK; NtoA ? ACK; goto TIME_WAIT;
	fi
}
active proctype peerB(){
	goto LISTEN;
CLOSED:
	if
	:: BtoN ! SYN; goto LISTEN;
	:: BtoN ! SYN; goto SYN_SENT;
	fi
CLOSE_WAIT:
	if
	:: BtoN ! FIN; goto CLOSING;
	:: NtoB ? FIN; BtoN ! ACK; NtoB ? ACK; goto CLOSE_WAIT;
	fi
CLOSING:
	if
	:: NtoB ? ACK; goto TIME_WAIT;
	:: NtoB ? FIN; BtoN ! ACK; NtoB ? ACK; goto CLOSING;
	fi
ESTABLISHED:
	if
	:: BtoN ! FIN; goto FIN_WAIT_1;
	:: NtoB ? FIN; BtoN ! ACK; NtoB ? ACK; goto CLOSE_WAIT;
	fi
FIN_WAIT_1:
	if
	:: NtoB ? ACK; goto FIN_WAIT_2;
	:: NtoB ? ACK; goto TIME_WAIT;
	fi
FIN_WAIT_2:
	if
	:: NtoB ? FIN; BtoN ! ACK; NtoB ? ACK; goto TIME_WAIT;
	fi
LAST_ACK:
	if
	:: NtoB ? ACK; goto CLOSED;
	:: NtoB ? FIN; BtoN ! ACK; NtoB ? ACK; goto LAST_ACK;
	fi
LISTEN:
	if
	:: BtoN ! SYN; goto SYN_SENT;
	fi
SYN_RECEIVED:
	if
	:: NtoB ? ACK; goto ESTABLISHED;
	:: NtoB ? FIN; BtoN ! ACK; NtoB ? ACK; goto CLOSE_WAIT;
	fi
SYN_SENT:
	if
	:: BtoN ! ACK; goto SYN_RECEIVED;
	fi
TIME_WAIT:
	if
	:: NtoB ? FIN; BtoN ! ACK; NtoB ? ACK; goto TIME_WAIT;
	fi
}
