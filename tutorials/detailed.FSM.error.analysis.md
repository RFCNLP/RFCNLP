# Detailed FSM Error Analysis

### GOLD TCP Errors

| Transition | Error Type | Reason | Text |
|--|--|--|--|
| `CLOSE_WAIT - FIN! -> LAST_ACK` | Missing | Target state is not explicit | **CLOSE-WAIT STATE**: Since the remote side **has already sent `FIN`**, RECEIVEs must be satisfied by text already on hand, but not yet delivered to the user. |
| `FIN_WAIT_1 - FIN?;ACK! -> CLOSING` | Missing | Text is ambiguous. Not clear how to handle otherwise statement | **FIN-WAIT-1**: **If our FIN has been ACKed** (perhaps in this segment), then enter TIME-WAIT, start the time-wait timer, turn off the other timers; **otherwise enter the CLOSING state.** |
| `LISTEN - SYN?;ACK! -> SYN_RECEIVED` | Missing | Partially correct transition that is eliminated by our filter, as we get  **SYN?;RST!;ACK!** instead. This is an extraction algorithm limitation, resulting from the incorrect handling of sibling/nested if-then statements. | Too long, see [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-annotated/TCP.txt#L4091) |
| `SYN_SENT - ACK?;ACK! -> ESTABLISHED` | Missing | Partially correct transition that is eliminated by our filter, as we get  **ACK?;RST!;RST?;SYN?;ACK!** instead. This is an extraction algorithm limitation, resulting from the incorrect handling of sibling/nested if-then statements. | Too long, see [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-annotated/TCP.txt#L4199) |
| `FIN_WAIT_1 - ACK? -> TIME_WAIT` | Incorrect | Text is ambiguous. Looks like a valid transition from the text. | **FIN-WAIT-1**: **If our FIN has been ACKed** (perhaps in this segment), then **enter TIME-WAIT**, start the time-wait timer, turn off the other timers; otherwise enter the CLOSING state. |
| `CLOSED - SYN? -> LISTEN` | Incorrect | Text is ambiguous + annotation error. Our expert did not annotate the OPEN event. The expected transition is (non-comm) **CLOSED - OPEN? -> LISTEN**. Additionally, there is an ambiguous mention of a **SYN?** | **CLOSED STATE** (i.e., TCB does not exist)  Note that some parts of the foreign socket may be unspecified in a **passive OPEN** and are to be filled in by the parameters of the **incoming SYN segment**. Verify the security and precedence requested are allowed for this user, if not return "error: precedence not allowed" or "error: security/compartment not allowed." **If passive, enter the LISTEN state and return**. |
| `TIME_WAIT - FIN?;ACK!;ACK? -> TIME_WAIT` | Partially Correct | Expected **FIN?;ACK!** instead. Extraction algorithm limitation resulting from the incorrect handling of sibling/nested if-then statements. An intermediate statement *"If our FIN has been ACKed"* is added to the transition, when it shouldn't be. | Too long, see [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-annotated/TCP.txt#L4091) |
| `SYN_RECEIVED - FIN?;ACK!;ACK? -> CLOSE_WAIT` | Partially Correct | Same as above | |
| `LAST_ACK - FIN?;ACK!;ACK? -> LAST_ACK` | Partially Correct | Same as above | |
| `FIN_WAIT_2 - FIN?;ACK!;ACK? -> TIME_WAIT` | Partially Correct | Same as above | |
| `ESTABLISHED - FIN?;ACK!;ACK? -> CLOSE_WAIT` | Partially Correct | Same as above | |
| `CLOSING - FIN?;ACK!;ACK? -> CLOSING`  | Partially Correct | Same as above | |
| `CLOSE_WAIT - FIN?;ACK!;ACK? -> CLOSE_WAIT` | Partially Correct | Same as above | |
| `SYN_SENT - ACK! -> SYN_RECEIVED` | Partially Correct |  Expected **SYN?;ACK** instead. This is a textual ambiguity issue. The expert annotated the send event as just ACK. There is no explicit mention of a receive action. | If  the  state  is  SYN-SENT  then  enter  SYN-RECEIVED,  form a SYN,ACK  segment  and  send  it. | 


### LinearCRF+R and Neural-CRF+R TCP Errors

*Both models produce the same FSM*

| Transition | Error Type | Reason | Text |
|--|--|--|--|
| `CLOSE_WAIT - FIN! -> LAST_ACK` | Missing | Same as Gold | |
| `LISTEN - SYN?;ACK! -> SYN_RECEIVED` | Missing | Same as Gold | |
| `FIN_WAIT_1 - FIN?;ACK! -> CLOSING` | Missing | Same as Gold | |
| `CLOSED - SYN? -> LISTEN` | Incorrect | Same as Gold | |
| `ESTABLISHED - FIN?;ACK!;ACK? -> CLOSING` | Incorrect | NLP prediction error. *Error* span predicted as *transition*. | See [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-predicted-paper/linear_phrases/TCP.xml#L799) | 
| `SYN_RECEIVED - FIN?;ACK!;ACK? -> CLOSING` | Incorrect | Same as above | |
| `FIN_WAIT_2 - SYN!;ACK! -> CLOSING` | Incorrect | NLP prediction error. *Error* span predicted as *transition*. | See [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-predicted-paper/linear_phrases/TCP.xml#L210) |
| `LAST_ACK - SYN!;ACK! -> CLOSING` | Incorrect | Same as above | |
| `TIME_WAIT - SYN!;ACK! -> CLOSING` | Incorrect | Same as above | |
| `CLOSING - SYN!;ACK! -> CLOSING` | Incorrect | Same as above | |
| `FIN_WAIT_1 - SYN!;ACK! -> CLOSING` | Incorrect | Same as above | |
| `LAST_ACK - ACK?;FIN! -> CLOSING` | Incorrect | NLP prediction error.  *Error* span predicted as *transition*. | See [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-predicted-paper/linear_phrases/TCP.xml#L276) | 
| `TIME_WAIT - ACK?;FIN! -> CLOSING` | Incorrect | Same as above | | 
| `CLOSING - ACK?;FIN! -> CLOSING` | Incorrect | Same as above | |
| `LISTEN - ACK?;FIN! -> ESTABLISHED` | Incorrect | No explicit event in the text, extraction algorithm looks in irrelevant block for an event. | **LISTEN state**, SYN-SENT state, SYN-RECEIVED state: queue for processing after **entering ESTABLISHED state.** |
| `TIME_WAIT - FIN?;ACK!;ACK? -> TIME_WAIT` | Partially Correct | Same as Gold | |
| `SYN_RECEIVED - FIN?;ACK!;ACK? -> CLOSE_WAIT` | Partially Correct | Same as Gold | |
| `LAST_ACK - FIN?;ACK!;ACK? -> LAST_ACK` | Partially Correct | Same as Gold  | |
| `FIN_WAIT_2 - FIN?;ACK!;ACK? -> TIME_WAIT` | Partially Correct | Same as Gold  | |
| `ESTABLISHED - FIN?;ACK!;ACK? -> CLOSE_WAIT` | Partially Correct | Same as Gold  | |
| `CLOSING - FIN?;ACK!;ACK? -> CLOSING`  | Partially Correct | Same as Gold  | |
| `CLOSE_WAIT - FIN?;ACK!;ACK? -> CLOSE_WAIT` | Partially Correct | Same as Gold  | |
| `SYN_SENT - SYN!;ACK! -> SYN_RECEIVED` | Partially Correct |  Same as Gold  | |
| `SYN_SENT - ACK?;FIN! -> ESTABLISHED` | Partially Correct. | Expected **ACK?;ACK!**. No explicit event in the text, extraction algorithm looks in irrelevant block for an event. This segment was not annotated as a transition in Gold.  | LISTEN state, **SYN-SENT state**, SYN-RECEIVED state: queue for processing after **entering ESTABLISHED state.** |
| `SYN_SENT - SYN!;ACK! -> ESTABLISHED` | Partially Correct  | Expected **ACK?;ACK!**. No explicit event in the text, extraction algorithm looks in irrelevant block for an event. This segment was not annotated as a transition in Gold. *Given that it introduces the same partially correct transition as above, it is not counted in stats.* | **SYN-SENT state**, SYN-RECEIVED state: Queue the data for transmission after **entering ESTABLISHED state.** |
| `SYN_RECEIVED - ACK?;FIN! -> ESTABLISHED`  | Partially Correct | Expected **ACK?**. No explicit event in the text, extraction algorithm looks in irrelevant block for an event. This segment was not annotated as a transition in Gold.  | LISTEN state, SYN-SENT state, **SYN-RECEIVED state**: queue for processing after **entering ESTABLISHED state.** |
`SYN_RECEIVED -  SYN!;ACK! -> ESTABLISHED` | Partially Correct | Expected **ACK?**. No explicit event in the text, extraction algorithm looks in irrelevant block for an event. This segment was not annotated as a transition in Gold. *Given that it introduces the same partially correct transition as above, it is not counted in stats.* | SYN-SENT state, **SYN-RECEIVED state**: Queue the data for transmission after **entering ESTABLISHED state.** |

### GOLD DCCP Errors

| Transition | Error Type | Reason | Text |
|--|--|--|--|
| `CLOSED - ε -> LISTEN` | Missing | Text ambiguity, expert knowledge is needed to infer the transition from the relevant text. | Each connection is actively initiated by one of the hosts, which we call the client; the other, initially passive host is called the server / **LISTEN**: Represents server sockets in the passive listening state. **LISTEN and CLOSED** are not associated with any particular DCCP connection. |
`CLOSING - DCCP-RESET? -> TIMEWAIT` | Missing | Text ambiguity, expert knowledge needed to understand what a handshake is and how that maps to expected client/server behaviors. We get the correct source and target state, but we parse a *timeout* event instead. |  DCCP connection termination uses a **handshake consisting of an optional DCCP-CloseReq packet, a DCCP-Close packet, and a DCCP-Reset packet**. The server moves from the OPEN state, possibly through the CLOSEREQ state, to CLOSED; **the client moves from OPEN through CLOSING to TIMEWAIT**, and after 2MSL wait time, to CLOSED |
`OPEN - DCCP-CLOSE! -> CLOSING` | Missing | Same as above | |
`OPEN - DCCP-CLOSEREQ?;DCCP-CLOSE! -> CLOSING` | Missing | Same as above |  |
`LISTEN - timeout -> CLOSED` | Missing | Not found in text | |
`OPEN - DCCP-ACK? -> OPEN` | Missing | Not found in text, only in ascii diagram | |
`OPEN - DCCP-CLOSE?;DCCP-RESET! -> CLOSED` | Missing | Source and target states are not explicit. | Either client or server may send a DCCP-Close packet, which will elicit a DCCP-Reset packet. |
`OPEN - DCCP-DATA! -> OPEN` | Missing | Not found in text | |
`OPEN - DCCP-DATA? -> OPEN` | Missing | Not found in text | |
`OPEN - DCCP-DATAACK! -> OPEN` | Missing | Not found in text | |
`OPEN - DCCP-DATAACK? -> OPEN` | Missing | Not found in text | |
`PARTOPEN - DCCP-CLOSE?;DCCP-RESET! -> CLOSED` | Missing | Text/indentation ambiguity. Cannot infer nested if-statement between PARTOPEN state in step 12 and transition in step 14. Expert knowledge is needed to know that “tear down connection” means transition to CLOSED. | Too long, see [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-annotated/DCCP.txt#L3972) |
`PARTOPEN - DCCP-CLOSEREQ?;DCCP-CLOSE! -> CLOSING` | Missing | Text/indentation ambiguity. Cannot infer nested if-statement between PARTOPEN state in step 12 and transition in step 13. | Too long, see [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-annotated/DCCP.txt#L3972) |
`PARTOPEN - DCCP-DATA?;DCCP-ACK! -> OPEN` | Missing | Annotation error. Our expert did not annotate this transition | DCCP connections' initiation phase consists of a three-way handshake: an initial DCCP-Request packet sent by the client, a DCCP-Response sent by the server in reply, and finally an acknowledgement from the client, **usually via a DCCP-Ack or DCCP-DataAck packet**. **The client moves from the REQUEST state to PARTOPEN, and finally to OPEN**; the server moves from LISTEN to RESPOND, and finally to OPEN. | 
`PARTOPEN - DCCP-DATAACK?;DCCP-ACK! -> OPEN` | Missing | Same as above | |
`REQUEST - DCCP-RESET? -> CLOSED` | Missing | No explicit mention of the source and target state | See [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-annotated/DCCP.txt#L3931) |
`REQUEST - DCCP-SYNC?;DCCP-RESET! -> CLOSED` | Missing | No explicit mention of the target state | **On receiving a sequence-valid DCCP-Sync packet**, ....  As an exception, if the peer endpoint is in the **REQUEST state, it MUST respond with a DCCP-Reset** instead of a DCCP-SyncAck. This serves to clean up DCCP A's half-open connection. |
`RESPOND - DCCP-DATAACK? -> OPEN` | Missing | No explicit mention of the event | **OPEN**: Client and server sockets **enter this state from PARTOPEN and RESPOND, respectively** | 
`CLOSING - timeout -> TIMEWAIT` | Incorrect | Text ambiguity, expert knowledge needed to understand what a handshake is and how that maps to expected client/server behaviors. Information is clarified in the ascii and examples of communication, which we do not consider. | DCCP connection termination uses a **handshake consisting of an optional DCCP-CloseReq packet, a DCCP-Close packet, and a DCCP-Reset packet**. The server moves from the OPEN state, possibly through the CLOSEREQ state, to CLOSED; **the client moves from OPEN through CLOSING to TIMEWAIT, and after 2MSL wait time, to CLOSED** |
`OPEN - timeout -> CLOSING` | Incorrect | Same as above | |
`PARTOPEN - DCCP-CLOSE? -> OPEN` | Incorrect | Text ambiguity, from the text it seems like transition would be valid. | **The client leaves the PARTOPEN state for OPEN** when it receives **a valid packet other than** DCCP-Response, DCCP-Reset, or DCCP-Sync from the server. |
`PARTOPEN - DCCP-CLOSEREQ? -> OPEN` | Incorrect | Same as above | |
`PARTOPEN - DCCP-SYNCACK? -> OPEN` | Incorrect | Same as above | |
`PARTOPEN - DCCP-RESET? -> OPEN` | Incorrect | Extraction algorithm error. It should handle "other than" correctly | **The client leaves the PARTOPEN state for OPEN** when it receives **a valid packet other than** DCCP-Response, DCCP-Reset, or DCCP-Sync from the server. |
`PARTOPEN - DCCP-RESPONSE? -> OPEN` | Incorrect | Same as above | |
`PARTOPEN - DCCP-SYNC? -> OPEN` | Incorrect | Same as above | |
| `PARTOPEN - DCCP-RESPONSE?;DCCP-ACK! -> PARTOPEN` | Partially correct | Expected **DCCP-ACK!** instead. Text ambiguity. Explicit mention of the response event is parsed | See [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-annotated/DCCP.txt#L3946) | 

*Note that just two spans of text in the RFC introduce all of the 8 incorrect transitions*

### LinearCRF+R DCCP Errors

| Transition | Error Type | Reason | Text |
|--|--|--|--|
| `CLOSED - ε -> LISTEN` | Missing | Same as Gold | |
| `CLOSING - DCCP-RESET? -> TIMEWAIT`  | Missing | Same as Gold | |
| `OPEN - DCCP-CLOSE! -> CLOSING`  | Missing | Same as Gold | |
| `OPEN - DCCP-CLOSEREQ?;DCCP-CLOSE!` | Missing | Same as Gold | |
| `LISTEN - timeout -> CLOSED`  | Missing | Same as Gold | |
| `OPEN - DCCP-ACK? -> OPEN` | Missing | Same as Gold | | 
| `OPEN - DCCP-CLOSE?;DCCP-RESET! -> CLOSED` | Missing | Same as Gold | |
| `OPEN - DCCP-DATA! -> OPEN` | Missing | Same as Gold | |
| `OPEN - DCCP-DATA? -> OPEN` | Missing | Same as Gold | |
| `OPEN - DCCP-DATAACK! -> OPEN` | Missing | Same as Gold | |
| `OPEN - DCCP-DATAACK? -> OPEN` | Missing | Same as Gold | |
| `PARTOPEN - DCCP-CLOSE?;DCCP-RESET! -> CLOSED` | Missing | Same as Gold | |
| `PARTOPEN - DCCP-CLOSEREQ?;DCCP-CLOSE! -> CLOSING` | Missing | Same as Gold | |
| `PARTOPEN - DCCP-DATA?;DCCP-ACK! -> OPEN` | Missing | Same as Gold | |
| `PARTOPEN - DCCP-DATAACK?;DCCP-ACK! -> OPEN` | Missing | Same as Gold | |
| `REQUEST - DCCP-RESET? -> CLOSED` | Missing | Same as Gold | |
| `REQUEST - DCCP-SYNC?;DCCP-RESET! -> CLOSED` | Missing | Same as Gold | |
| `RESPOND - DCCP-DATAACK? -> OPEN` | Missing | Same as Gold | |
| `CLOSEREQ - DCCP-CLOSE?;DCCP-RESET! -> CLOSED` | Missing | Textual ambiguity, expert knowledge needed to know what a handshake is and how it maps to server/client behavior. We get the correct states, but parse message as **DCCP-CLOSEREQ?** instead | DCCP connection termination uses a **handshake consisting of an optional DCCP-CloseReq packet, a DCCP-Close packet, and a DCCP-Reset packet**. The server **moves from the OPEN state, possibly through the CLOSEREQ state, to CLOSED**; the client moves from OPEN through CLOSING to TIMEWAIT, and after 2MSL wait time, to CLOSED |
| `PARTOPEN - DCCP-ACK? -> OPEN` | Missing | NLP prediciton error. Not all events that are enumerated after *"other than"* statement are predicted in action span. | See [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-predicted-paper/linear_phrases/DCCP.xml#L539) |
| `PARTOPEN - DCCP-DATA? -> OPEN` | Missing | Same as above | |
| `PARTOPEN - DCCP-DATAACK? -> OPEN` | Missing | Same as above | |
| `PARTOPEN - DCCP-REQUEST? -> OPEN` | Missing | Same as above | |
| `PARTOPEN - DCCP-DATAACK! -> PARTOPEN` | Missing | NLP prediction error. *Transition* span is marked as *action* | See [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-predicted-paper/linear_phrases/DCCP.xml#L504) |
| `PARTOPEN - timeout -> CLOSED` | Missing | NLP prediction error. This is a very hard case, as the target state is not explicit. In this case, the wording *“reset the connection...”* was explicitly annotated by the expert in GOLD as a transition to CLOSED, which was not done in other cases. | If the client remains in PARTOPEN for more than 4MSL, (8 minutes), **it SHOULD reset the connection with Reset Code 2, "Aborted".** |
|  `TIMEWAIT - timeout -> CLOSED` | Missing | NLP prediction error. *Trigger* span predicted as *action* | See [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-predicted-paper/linear_phrases/DCCP.xml#L561) |
| `CLOSEREQ - DCCP-CLOSEREQ! -> CLOSED` | Incorrect | Textual ambiguity, expert knowledge needed to know what a handshake is and how it maps to server/client behavior. | DCCP connection termination uses a **handshake consisting of an optional DCCP-CloseReq packet, a DCCP-Close packet, and a DCCP-Reset packet**. The server **moves from the OPEN state, possibly through the CLOSEREQ state, to CLOSED**; the client moves from OPEN through CLOSING to TIMEWAIT, and after 2MSL wait time, to CLOSED |
| `CLOSEREQ - ε -> TIMEWAIT` | Incorrect | NLP prediction error. *Transition* span predicted as *action* | See [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-predicted-paper/linear_phrases/DCCP.xml#L56) | 
| `CLOSING - timeout -> TIMEWAIT` | Incorrect | Extraction algorithm error. Incorrect handling of nested if-then statements, doesn’t find the right event  | See [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-predicted-paper/linear_phrases/DCCP.xml#L566) |
| `OPEN - timeout -> CLOSING` | Incorrect | Same as above | | 
| `PARTOPEN - DCCP-RESPONSE? -> OPEN` | Incorrect | NLP prediction error. *Trigger* span is predicted as *transition* | See [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-predicted-paper/linear_phrases/DCCP.xml#L532) | 
| `PARTOPEN - DCCP-RESPONSE?;DCCP-DATA! -> REQUEST` | Incorrect | NLP post-processing error. Source and target state are swapped | See [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-predicted-paper/linear_phrases/DCCP.xml#L39) | 
| `TIMEWAIT - DCCP-RESET? -> TIMEWAIT` | Incorrect | NLP prediction error. *Timer* span is predicted as *trigger* | See [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-predicted-paper/linear_phrases/DCCP.xml#L757) |
| `TIMEWAIT - timeout -> TIMEWAIT` | Incorrect | NLP prediction error. Outside spans predicted as transitions. Really hard case, as there is transition-style language. | See [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-predicted-paper/linear_phrases/DCCP.xml#L66) | 
| `UNSTABLE - ε -> RESPOND` | Incorrect | Extraction algorithm error. It cannot find a source state and looks in irrelevant block | See [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-predicted-paper/linear_phrases/DCCP.xml#L233) |
| `LISTEN - DCCP-REQUEST? -> RESPOND` | Partially Correct | Expected **DCCP-REQUEST?;DCCP-RESPONSE!** instead. This transition is predicted in a few places. In the first mention, the prediction looks good as the response event is not explicit. In the second and third mentions, it seems like it should have been extracted based on NLP prediction, so it is likely an algorithm extraction error | RESPOND: A server socket enters this state, from LISTEN, after receiving a DCCP-Request from a client. Also see [here](https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-predicted-paper/linear_phrases/DCCP.xml#L659) |
| `PARTOPEN - DCCP-ACK!;timeout;DCCP-RESET? -> PARTOPEN` | Partially correct | Same as Gold | | 

### NeuralCRF+R DCCP Errors

| Transition | Error Type | Reason | Text |
|--|--|--|--|
|--|--|--|--|
| `CLOSED - ε -> LISTEN` | Missing | Same as Gold | |
| `CLOSING - DCCP-RESET? -> TIMEWAIT`  | Missing | Same as Gold | |
| `OPEN - DCCP-CLOSE! -> CLOSING`  | Missing | Same as Gold | |
| `OPEN - DCCP-CLOSEREQ?;DCCP-CLOSE! -> CLOSING` | Missing | Same as Gold | |
| `LISTEN - timeout -> CLOSED`  | Missing | Same as Gold | |
| `OPEN - DCCP-ACK? -> OPEN` | Missing | Same as Gold | | 
| `OPEN - DCCP-CLOSE?;DCCP-RESET! -> CLOSED` | Missing | Same as Gold | |
| `OPEN - DCCP-DATA! -> OPEN` | Missing | Same as Gold | |
| `OPEN - DCCP-DATA? -> OPEN` | Missing | Same as Gold | |
| `OPEN - DCCP-DATAACK! -> OPEN` | Missing | Same as Gold | |
| `OPEN - DCCP-DATAACK? -> OPEN` | Missing | Same as Gold | |
| `PARTOPEN - DCCP-CLOSE?;DCCP-RESET! -> CLOSED` | Missing | Same as Gold | |
| `PARTOPEN - DCCP-CLOSEREQ?;DCCP-CLOSE! -> CLOSING` | Missing | Same as Gold | |
| `PARTOPEN - DCCP-DATA?;DCCP-ACK! -> OPEN` | Missing | Same as Gold | |
| `PARTOPEN - DCCP-DATAACK?;DCCP-ACK! -> OPEN` | Missing | Same as Gold | |
| `REQUEST - DCCP-RESET? -> CLOSED` | Missing | Same as Gold | |
| `REQUEST - DCCP-SYNC?;DCCP-RESET! -> CLOSED` | Missing | Same as Gold | |
| `RESPOND - DCCP-DATAACK? -> OPEN` | Missing | Same as Gold | |
| `CLOSEREQ - DCCP-CLOSE?;DCCP-RESET! -> CLOSED` | Missing | Same as Linear | |
| `PARTOPEN - DCCP-ACK? -> OPEN` | Missing | Same as Linear | |
| `PARTOPEN - DCCP-DATA? -> OPEN` | Missing | Same as Linear | |
| `PARTOPEN - DCCP-DATAACK? -> OPEN` | Missing | Same as Linear | |
| `PARTOPEN - DCCP-REQUEST? -> OPEN` | Missing | Same as Linear | |
| `PARTOPEN - DCCP-DATAACK! -> PARTOPEN` | Missing | Same as Linear | |
| `PARTOPEN - timeout -> CLOSED` | Missing | Same as Linear | |
| `TIMEWAIT - timeout -> CLOSED` | Missing | Same as Linear | |
| `CLOSEREQ - ε -> TIMEWAIT` | Incorrect | Same as Linear | |
| `CLOSING - timeout -> TIMEWAIT` | Incorrect | Same as Linear | | 
| `OPEN - timeout -> CLOSING` | Incorrect | Same as Linear | | 
| `PARTOPEN - DCCP-RESPONSE?;DCCP-DATA! -> REQUEST` | Incorrect | Same as Linear | | 
| `TIMEWAIT - DCCP-RESET? -> TIMEWAIT` | Incorrect | Same as Linear | | 
| `TIMEWAIT - timeout -> TIMEWAIT` | Incorrect | Same as Linear | | 
| `UNSTABLE - ε -> RESPOND` | Incorrect | Same as Linear | | 
| `PARTOPEN - DCCP-RESPONSE?;DCCP-RESET?;DCCP-SYNC? -> OPEN` | Incorrect | NLP prediction error. *Action* spans predicted as *trigger* | See https://github.com/RFCNLP/RFCNLP/blob/main/rfcs-predicted-paper/bert_pretrained_rfcs_crf_phrases_feats/DCCP.xml#L509 |
| `LISTEN - DCCP-REQUEST? -> RESPOND` | Partially Correct | Same as Linear | |
| `PARTOPEN - DCCP-ACK!;timeout;DCCP-RESET? -> PARTOPEN` | Partially correct | Same as Gold | | 