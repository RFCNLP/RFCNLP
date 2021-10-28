ltl phi3 {
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
}