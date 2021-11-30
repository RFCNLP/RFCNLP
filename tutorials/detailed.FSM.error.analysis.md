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
