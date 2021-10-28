ltl phi4 {
	always (
		!(
			state[0] == CloseReqState &&
			state[1] == CloseReqState
		)
	)
}
