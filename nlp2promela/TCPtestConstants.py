# coding: utf-8
'''
name       : TCPtestConstants.py
author     : [redacted]
authored   : 6 May 2021
description: test constants for TCP
'''
import glob

canonical_TCP_model = "promela-models/TCP/Canonical-Test/Canonical-TCP-test.pml"
canonical_TCP_IO    = "promela-models/TCP/korg-components/IO.txt"
TCP_props_path      = "promela-models/TCP/props/"
TCP_props           = [a for a in glob.glob(TCP_props_path + "phi*.pml")]

TCP_AP_prefix = """
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
"""