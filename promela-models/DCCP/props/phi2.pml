ltl phi2 {
	always (
        !(state[0] == TimeWaitState && state[1] == TimeWaitState)
    )
}