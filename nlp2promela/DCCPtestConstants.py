# coding: utf-8
'''
name       : DCCPtestConstants.py
author     : [redacted]
authored   : 6 May 2021
description: test constants for TCP
'''
import glob

# canonical_DCCP_model = "promela-models/DCCP/Canonical-Test/Canonical-DCCP-test.pml"

canonical_DCCP_model = "promela-models/DCCP/Canonical-Test-Client-Server/" + \
                       "canonical.DCCP.client.server.pml"

canonical_DCCP_IO    = "promela-models/DCCP/korg-components/IO.txt"
DCCP_props_path      = "promela-models/DCCP/props/"
DCCP_props           = [a for a in glob.glob(DCCP_props_path + "phi*.pml")]

DCCP_AP_prefix = """
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
"""