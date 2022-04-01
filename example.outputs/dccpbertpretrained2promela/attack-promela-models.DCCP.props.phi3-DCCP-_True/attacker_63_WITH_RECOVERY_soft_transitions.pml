/* spin -t63 -s -r -p -g attack-promela-models.DCCP.props.phi3-DCCP-_daisy_check.pml */
active proctype attacker() {
	
	if
	:: BtoN ? DCCP_REQUEST;
	fi unless timeout;
	if
	:: NtoB ! DCCP_RESPONSE;
	fi unless timeout;
	if
	:: BtoN ? DCCP_ACK;
	fi unless timeout;
	if
	:: NtoB ! DCCP_DATA;
	fi unless timeout;
	if
	:: NtoB ! DCCP_DATA;
	fi unless timeout;
	if
	:: NtoB ! DCCP_DATA;
	fi unless timeout;
	if
	:: BtoN ? DCCP_CLOSEREQ;
	fi unless timeout;
	if
	:: NtoB ! DCCP_CLOSEREQ;
	fi unless timeout;
	if
	:: NtoB ! DCCP_RESET;
	fi unless timeout;
	if
	:: NtoB ! DCCP_RESET;
	fi unless timeout;
	if
	:: AtoN ? DCCP_REQUEST;
	fi unless timeout;
	if
	:: NtoB ! DCCP_RESET;
	fi unless timeout;
// recovery to N
// N begins here ... 

	do
	:: AtoN ? DCCP_REQUEST -> 
		if
		:: NtoB ! DCCP_REQUEST;
		fi unless timeout;

	:: AtoN ? DCCP_RESPONSE -> 
		if
		:: NtoB ! DCCP_RESPONSE;
		fi unless timeout;

	:: AtoN ? DCCP_DATA -> 
		if
		:: NtoB ! DCCP_DATA;
		fi unless timeout;

	:: AtoN ? DCCP_ACK -> 
		if
		:: NtoB ! DCCP_ACK;
		fi unless timeout;

	:: AtoN ? DCCP_DATAACK -> 
		if
		:: NtoB ! DCCP_DATAACK;
		fi unless timeout;

	:: AtoN ? DCCP_CLOSEREQ -> 
		if
		:: NtoB ! DCCP_CLOSEREQ;
		fi unless timeout;

	:: AtoN ? DCCP_CLOSE -> 
		if
		:: NtoB ! DCCP_CLOSE;
		fi unless timeout;

	:: AtoN ? DCCP_RESET -> 
		if
		:: NtoB ! DCCP_RESET;
		fi unless timeout;

	:: AtoN ? DCCP_SYNC -> 
		if
		:: NtoB ! DCCP_SYNC;
		fi unless timeout;

	:: AtoN ? DCCP_SYNCACK -> 
		if
		:: NtoB ! DCCP_SYNCACK;
		fi unless timeout;

	:: BtoN ? DCCP_REQUEST -> 
		if
		:: NtoA ! DCCP_REQUEST;
		fi unless timeout;

	:: BtoN ? DCCP_RESPONSE -> 
		if
		:: NtoA ! DCCP_RESPONSE;
		fi unless timeout;

	:: BtoN ? DCCP_DATA -> 
		if
		:: NtoA ! DCCP_DATA;
		fi unless timeout;

	:: BtoN ? DCCP_ACK -> 
		if
		:: NtoA ! DCCP_ACK;
		fi unless timeout;

	:: BtoN ? DCCP_DATAACK -> 
		if
		:: NtoA ! DCCP_DATAACK;
		fi unless timeout;

	:: BtoN ? DCCP_CLOSEREQ -> 
		if
		:: NtoA ! DCCP_CLOSEREQ;
		fi unless timeout;

	:: BtoN ? DCCP_CLOSE -> 
		if
		:: NtoA ! DCCP_CLOSE;
		fi unless timeout;

	:: BtoN ? DCCP_RESET -> 
		if
		:: NtoA ! DCCP_RESET;
		fi unless timeout;

	:: BtoN ? DCCP_SYNC -> 
		if
		:: NtoA ! DCCP_SYNC;
		fi unless timeout;

	:: BtoN ? DCCP_SYNCACK -> 
		if
		:: NtoA ! DCCP_SYNCACK;
		fi unless timeout;

	od

}