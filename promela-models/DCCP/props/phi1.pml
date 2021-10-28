ltl phi1 {
    ! ( eventually 
        always (
            (state[0] == before_state[0]) &&
            (state[1] == before_state[1])
        )
    )
}