ltl phi2 {
	( (always ( eventually ( state[0] == 1 && state[1] == 2 ) ) ) 
		implies ( eventually ( state[0] == 4 ) ) )
}