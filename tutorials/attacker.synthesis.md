# Attacker Synthesis Tutorial

When we run an attacker synthesis target, e.g.:

```
make tcp2promela
```

what happens in the background is a call to `nlp2promela.py`, e.g.:

```
python3 nlp2promela/nlp2promela.py rfcs-annotated-tidied/TCP.xml
```

As described in the [README](../README.md), the script saves an image file and a Promela program, each of which captures the extracted FSM.

In the special case where the input is an intermediary representation derived from the TCP or DCCP RFC, our code takes some additional steps, namely:
1. compare the extracted FSM to a canonical FSM;
2. check which properties (from a list of known correctness properties) the extracted FSM supports; and
3. perform automated attacker synthesis on the extracted FSM.

Our code can be used to perform attacker synthesis against other RFCs, but you need to write correctness properties first.  The process is a little complicated - we will document how to do this in the future.

The CLI output begins with a comparison of the extracted FSM to the canonical FSM.  For example:

```
	CORRECT STATES:

		SYN_SENT, TIME_WAIT, SYN_RECEIVED, LAST_ACK, CLOSED, ESTABLISHED, CLOSING,
		FIN_WAIT_1, FIN_WAIT_2, LISTEN, CLOSE_WAIT

	WRONG STATES: ∅

	MISSING STATES: ∅


~~~~~~~~~~~~~~~~~~~~~~~~~ 19 CORRECT TRANSITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~


		8 Correct Communication Transitions
╒══════════════╤═════════╤═══════════════╤════════════════════════════════╕
│ Source       │ Label   │ Destination   │ Line #s                        │
╞══════════════╪═════════╪═══════════════╪════════════════════════════════╡
│ CLOSED       │ SYN!    │ SYN_SENT      │ [2078] ---> [3384]             │
├──────────────┼─────────┼───────────────┼────────────────────────────────┤
│ CLOSE_WAIT   │ FIN!    │ CLOSING       │ [2239] ---> [3784]             │
├──────────────┼─────────┼───────────────┼────────────────────────────────┤
│ CLOSING      │ ACK?    │ TIME_WAIT     │ [2607] ---> [4507]             │
├──────────────┼─────────┼───────────────┼────────────────────────────────┤
│ ESTABLISHED  │ FIN!    │ FIN_WAIT_1    │ [2220, 2224] ---> [3743, 3749] │
├──────────────┼─────────┼───────────────┼────────────────────────────────┤
│ FIN_WAIT_1   │ ACK?    │ FIN_WAIT_2    │ [2597] ---> [4491]             │
├──────────────┼─────────┼───────────────┼────────────────────────────────┤
│ LAST_ACK     │ ACK?    │ CLOSED        │ [2612] ---> [4515]             │
├──────────────┼─────────┼───────────────┼────────────────────────────────┤
│ LISTEN       │ SYN!    │ SYN_SENT      │ [2087, 2123] ---> [3396, 3498] │
├──────────────┼─────────┼───────────────┼────────────────────────────────┤
│ SYN_RECEIVED │ ACK?    │ ESTABLISHED   │ [2565] ---> [4437]             │
╘══════════════╧═════════╧═══════════════╧════════════════════════════════╛

		0 Correct User Call Transitions
╒══════════╤═════════╤═══════════════╤═══════════╕
│ Source   │ Label   │ Destination   │ Line #s   │
╞══════════╪═════════╪═══════════════╪═══════════╡
╘══════════╧═════════╧═══════════════╧═══════════╛

		11 Correct Reset Transitions
╒══════════════╤═════════╤═══════════════╤════════════════════════════════════════════╕
│ Source       │ Label   │ Destination   │ Line #s                                    │
╞══════════════╪═════════╪═══════════════╪════════════════════════════════════════════╡
│ CLOSE_WAIT   │ RST?    │ CLOSED        │ [2510] ---> [4343]                         │
├──────────────┼─────────┼───────────────┼────────────────────────────────────────────┤
│ CLOSING      │ RST?    │ CLOSED        │ [2515] ---> [4350]                         │
├──────────────┼─────────┼───────────────┼────────────────────────────────────────────┤
│ ESTABLISHED  │ RST?    │ CLOSED        │ [2510] ---> [4343]                         │
├──────────────┼─────────┼───────────────┼────────────────────────────────────────────┤
│ FIN_WAIT_1   │ RST?    │ CLOSED        │ [2510] ---> [4343]                         │
├──────────────┼─────────┼───────────────┼────────────────────────────────────────────┤
│ FIN_WAIT_2   │ RST?    │ CLOSED        │ [2510] ---> [4343]                         │
├──────────────┼─────────┼───────────────┼────────────────────────────────────────────┤
│ LAST_ACK     │ RST?    │ CLOSED        │ [2515] ---> [4350]                         │
├──────────────┼─────────┼───────────────┼────────────────────────────────────────────┤
│ LISTEN       │ RST?    │ CLOSED        │ [1424, 1426, 2501] ---> [2389, 2392, 4332] │
├──────────────┼─────────┼───────────────┼────────────────────────────────────────────┤
│ LISTEN       │ RST?    │ LISTEN        │ [1423, 2495] ---> [2388, 4326]             │
├──────────────┼─────────┼───────────────┼────────────────────────────────────────────┤
│ SYN_RECEIVED │ RST?    │ CLOSED        │ [1424, 1426] ---> [2389, 2392]             │
├──────────────┼─────────┼───────────────┼────────────────────────────────────────────┤
│ SYN_RECEIVED │ RST?    │ LISTEN        │ [1423] ---> [2388]                         │
├──────────────┼─────────┼───────────────┼────────────────────────────────────────────┤
│ TIME_WAIT    │ RST?    │ CLOSED        │ [2515] ---> [4350]                         │
╘══════════════╧═════════╧═══════════════╧════════════════════════════════════════════╛


		27 WRONG TRANSITIONS, of which 10 pass the filter

╒══════════════╤══════════════════════════╤═══════════════╤════════════════════╤══════════════════════════════════════════════════════════════════════════════╤══════════════════╕
│ Source       │ Label                    │ Destination   │ Line #s            │ Diagnosis                                                                    │ Passes Filter?   │
╞══════════════╪══════════════════════════╪═══════════════╪════════════════════╪══════════════════════════════════════════════════════════════════════════════╪══════════════════╡
│ CLOSED       │ SYN;SYN!                 │ LISTEN        │ [2073] ---> [3379] │ SWAP ARG W/ SOME l ∈ ['OPEN?'] - see: [(User)]                               │ Yes              │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ CLOSE_WAIT   │ FIN?;ACK!;ACK?           │ CLOSE_WAIT    │ [2690] ---> [4645] │ SWAP ARG W/ SOME l ∈ ['FIN?;ACK!'] - see: [(Comm)]                           │ Yes              │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ CLOSE_WAIT   │ SYN?;RST!                │ CLOSED        │ [2551] ---> [4407] │ SWAP ARG W/ SOME l ∈ ['RST?', 'ABORT?;RST!'] - see: [(RST),(User)]           │ No               │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ CLOSING      │ ABORT?;RST!              │ CLOSED        │ [2275] ---> [3878] │ SWAP ARG W/ SOME l ∈ ['ABORT?', 'RST?'] - see: [(RST),(User)]                │ No               │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ CLOSING      │ FIN?;ACK!;ACK?           │ CLOSING       │ [2692] ---> [4649] │ SWAP ARG W/ SOME l ∈ ['FIN?;ACK!'] - see: [(Comm)]                           │ Yes              │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ CLOSING      │ SYN?;RST!                │ CLOSED        │ [2551] ---> [4407] │ SWAP ARG W/ SOME l ∈ ['ABORT?', 'RST?'] - see: [(RST),(User)]                │ No               │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ ESTABLISHED  │ FIN?;ACK!;ACK?           │ CLOSE_WAIT    │ [2681] ---> [4630] │ SWAP ARG W/ SOME l ∈ ['FIN?;ACK!'] - see: [(Comm)]                           │ Yes              │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ ESTABLISHED  │ SYN?;RST!                │ CLOSED        │ [2551] ---> [4407] │ SWAP ARG W/ SOME l ∈ ['RST?', 'ABORT?;RST!'] - see: [(RST),(User)]           │ No               │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ FIN_WAIT_1   │ ACK?                     │ TIME_WAIT     │ [2684] ---> [4634] │ SWAP START W/ SOME x ∈ ['CLOSING'] - see: [(Comm)]                           │ Yes              │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ FIN_WAIT_1   │ RST!                     │ CLOSED        │ [2271] ---> [3873] │ SWAP ARG W/ SOME l ∈ ['ABORT?;RST!', 'RST?'] - see: [(RST),(User)]           │ No               │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ FIN_WAIT_2   │ FIN?;ACK!;ACK?           │ TIME_WAIT     │ [2687] ---> [4639] │ SWAP ARG W/ SOME l ∈ ['FIN?;ACK!'] - see: [(Comm)]                           │ Yes              │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ FIN_WAIT_2   │ RST!                     │ CLOSED        │ [2271] ---> [3873] │ SWAP ARG W/ SOME l ∈ ['ABORT?;RST!', 'RST?'] - see: [(RST),(User)]           │ No               │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ LAST_ACK     │ ABORT?;RST!              │ CLOSED        │ [2275] ---> [3878] │ SWAP ARG W/ SOME l ∈ ['ABORT?', 'ACK?', 'RST?'] - see: [(RST),(Comm),(User)] │ No               │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ LAST_ACK     │ FIN?;ACK!;ACK?           │ LAST_ACK      │ [2694] ---> [4654] │ SWAP ARG W/ SOME l ∈ ['FIN?;ACK!'] - see: [(Comm)]                           │ Yes              │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ LAST_ACK     │ SYN?;RST!                │ CLOSED        │ [2551] ---> [4407] │ SWAP ARG W/ SOME l ∈ ['ABORT?', 'ACK?', 'RST?'] - see: [(RST),(Comm),(User)] │ No               │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ LISTEN       │ ABORT?;RST!              │ CLOSED        │ [2256] ---> [3851] │ SWAP ARG W/ SOME l ∈ ['RST?', 'ABORT?', 'CLOSE?'] - see: [(RST),(User)]      │ No               │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ LISTEN       │ CLOSE?;FIN!              │ CLOSED        │ [2214] ---> [3734] │ SWAP ARG W/ SOME l ∈ ['RST?', 'ABORT?', 'CLOSE?'] - see: [(RST),(User)]      │ No               │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ LISTEN       │ SYN?;RST!;ACK!           │ SYN_RECEIVED  │ [2361] ---> [4090] │ SWAP ARG W/ SOME l ∈ ['SYN?;ACK!'] - see: [(Comm)]                           │ No               │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ SYN_RECEIVED │ FIN?;ACK!;ACK?           │ CLOSE_WAIT    │ [2681] ---> [4630] │ SWAP ARG W/ SOME l ∈ ['FIN?;ACK!'] - see: [(Comm)]                           │ Yes              │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ SYN_RECEIVED │ SYN?;RST!                │ CLOSED        │ [2551] ---> [4407] │ SWAP ARG W/ SOME l ∈ ['RST?', 'ABORT?;RST!'] - see: [(RST),(User)]           │ No               │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ SYN_SENT     │ ABORT?;RST!              │ CLOSED        │ [2260] ---> [3858] │ SWAP ARG W/ SOME l ∈ ['ACK?;RST?', 'ABORT?', 'CLOSE?'] - see: [(RST),(User)] │ No               │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ SYN_SENT     │ ACK!                     │ SYN_RECEIVED  │ [2433] ---> [4209] │ SWAP ARG W/ SOME l ∈ ['SYN?;ACK!'] - see: [(Comm)]                           │ Yes              │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ SYN_SENT     │ ACK?;RST!;RST?;SYN?;ACK! │ ESTABLISHED   │ [2426] ---> [4198] │ SWAP ARG W/ SOME l ∈ ['ACK?;ACK!'] - see: [(Comm)]                           │ No               │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ SYN_SENT     │ RST?;ACK?                │ CLOSED        │ [2390] ---> [4137] │ SWAP ARG W/ SOME l ∈ ['ACK?;RST?', 'ABORT?', 'CLOSE?'] - see: [(RST),(User)] │ No               │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ TIME_WAIT    │ ABORT?;RST!              │ CLOSED        │ [2275] ---> [3878] │ SWAP ARG W/ SOME l ∈ ['ABORT?', 'RST?'] - see: [(RST),(User)]                │ No               │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ TIME_WAIT    │ FIN?;ACK!;ACK?           │ TIME_WAIT     │ [2701] ---> [4667] │ SWAP ARG W/ SOME l ∈ ['FIN?;ACK!'] - see: [(Comm)]                           │ Yes              │
├──────────────┼──────────────────────────┼───────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────┤
│ TIME_WAIT    │ SYN?;RST!                │ CLOSED        │ [2551] ---> [4407] │ SWAP ARG W/ SOME l ∈ ['ABORT?', 'RST?'] - see: [(RST),(User)]                │ No               │
╘══════════════╧══════════════════════════╧═══════════════╧════════════════════╧══════════════════════════════════════════════════════════════════════════════╧══════════════════╛


~~~~~~~~~~~~~~~~~~~~~~~~~ 29 MISSING TRANSITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~ 


		12 Missing Communication Transitions
╒══════════════╤═══════════╤═══════════════╕
│ Source       │ Label     │ Destination   │
╞══════════════╪═══════════╪═══════════════╡
│ CLOSE_WAIT   │ FIN!      │ LAST_ACK      │
├──────────────┼───────────┼───────────────┤
│ CLOSE_WAIT   │ FIN?;ACK! │ CLOSE_WAIT    │
├──────────────┼───────────┼───────────────┤
│ CLOSING      │ FIN?;ACK! │ CLOSING       │
├──────────────┼───────────┼───────────────┤
│ ESTABLISHED  │ FIN?;ACK! │ CLOSE_WAIT    │
├──────────────┼───────────┼───────────────┤
│ FIN_WAIT_1   │ FIN?;ACK! │ CLOSING       │
├──────────────┼───────────┼───────────────┤
│ FIN_WAIT_2   │ FIN?;ACK! │ TIME_WAIT     │
├──────────────┼───────────┼───────────────┤
│ LAST_ACK     │ FIN?;ACK! │ LAST_ACK      │
├──────────────┼───────────┼───────────────┤
│ LISTEN       │ SYN?;ACK! │ SYN_RECEIVED  │
├──────────────┼───────────┼───────────────┤
│ SYN_RECEIVED │ FIN?;ACK! │ CLOSE_WAIT    │
├──────────────┼───────────┼───────────────┤
│ SYN_SENT     │ ACK?;ACK! │ ESTABLISHED   │
├──────────────┼───────────┼───────────────┤
│ SYN_SENT     │ SYN?;ACK! │ SYN_RECEIVED  │
├──────────────┼───────────┼───────────────┤
│ TIME_WAIT    │ FIN?;ACK! │ TIME_WAIT     │
╘══════════════╧═══════════╧═══════════════╛

		16 Missing User Call Transitions
╒══════════════╤═════════════╤═══════════════╕
│ Source       │ Label       │ Destination   │
╞══════════════╪═════════════╪═══════════════╡
│ CLOSED       │ OPEN?       │ LISTEN        │
├──────────────┼─────────────┼───────────────┤
│ CLOSE_WAIT   │ ABORT?;RST! │ CLOSED        │
├──────────────┼─────────────┼───────────────┤
│ CLOSE_WAIT   │ CLOSE?;FIN! │ CLOSING       │
├──────────────┼─────────────┼───────────────┤
│ CLOSING      │ ABORT?      │ CLOSED        │
├──────────────┼─────────────┼───────────────┤
│ ESTABLISHED  │ ABORT?;RST! │ CLOSED        │
├──────────────┼─────────────┼───────────────┤
│ ESTABLISHED  │ CLOSE?;FIN! │ FIN_WAIT_1    │
├──────────────┼─────────────┼───────────────┤
│ FIN_WAIT_1   │ ABORT?;RST! │ CLOSED        │
├──────────────┼─────────────┼───────────────┤
│ FIN_WAIT_2   │ ABORT?;RST! │ CLOSED        │
├──────────────┼─────────────┼───────────────┤
│ LAST_ACK     │ ABORT?      │ CLOSED        │
├──────────────┼─────────────┼───────────────┤
│ LISTEN       │ ABORT?      │ CLOSED        │
├──────────────┼─────────────┼───────────────┤
│ LISTEN       │ CLOSE?      │ CLOSED        │
├──────────────┼─────────────┼───────────────┤
│ SYN_RECEIVED │ ABORT?;RST! │ CLOSED        │
├──────────────┼─────────────┼───────────────┤
│ SYN_RECEIVED │ CLOSE?;FIN! │ FIN_WAIT_1    │
├──────────────┼─────────────┼───────────────┤
│ SYN_SENT     │ ABORT?      │ CLOSED        │
├──────────────┼─────────────┼───────────────┤
│ SYN_SENT     │ CLOSE?      │ CLOSED        │
├──────────────┼─────────────┼───────────────┤
│ TIME_WAIT    │ ABORT?      │ CLOSED        │
╘══════════════╧═════════════╧═══════════════╛

		1 Missing Reset Transitions
╒══════════╤═══════════╤═══════════════╕
│ Source   │ Label     │ Destination   │
╞══════════╪═══════════╪═══════════════╡
│ SYN_SENT │ ACK?;RST? │ CLOSED        │
╘══════════╧═══════════╧═══════════════╛

~~~~~~~~~~~~~~~~~~~~~~~~~~ SUMMARY STATISTICS ~~~~~~~~~~~~~~~~~~~~~~~~~~


We expect 20 Communication transitions.

	Of those, we find 8 but are still missing 12.

We expect 16 User Calls transitions.

	Of those, we find 0 but are still missing 16.

We expect 12 Resets transitions.

	Of those, we find 11 but are still missing 1.

```

These results are reported in Table III of our paper.  We only report the Communication transitions, because for the properties we study, they are the only ones that matter.

Next, the tool uses the Spin model checker to determine which properties are supported by the extracted FSM (as a Promela program), and which are not.  That output looks like, e.g.:

```
+++++++++++++++++++++++++ SUPPORTED PROPERTIES +++++++++++++++++++++++++

WROTE TO TEMPORARY-net-rem-7326678231000585354/288689041877425805.pml
WROTE TO TEMPORARY-net-rem-7326678231000585354/4934667472316810343.pml
TCP.pml⊨promela-models/TCP/props/phi1.pml
TCP.pml⊨promela-models/TCP/props/phi2.pml
TCP.pml⊭promela-models/TCP/props/phi3.pml
TCP.pml⊭promela-models/TCP/props/phi5.pml
```

Basically, if the Promela program `P` supports the property `phi`, we write `P⊨phi`; if it does not, then we write `P⊭phi` (notice that `⊭` is like `⊨` but with a line through it).  We make this a little easier to read in the terminal output by coloring the "supports" symbol `⊨` green, and the "does not support" symbol `⊭` red/yellow/orange (depending on your terminal theme).  

If you are unfamiliar with the relevant formal methods, the word "supports" might seem strange -- you can interpret it as meaning "verifies", "makes true", etc.  Basically, `P⊨phi` means `phi` is a property which is true about the Promela program `P`.

These results are reported in Table IV of our paper.

Next, we synthesize condidate attackers using each of the supported properties.  This is probably the most confusing part of the terminal output.  First, you will see output from Spin at multiple stages of this process, e.g.:

```
============ TRY TO ATTACK promela-models/TCP/props/phi1.pml============
remainder = TEMPORARY-net-rem-9222408737716169234/7095940097103984140.pml
prop = promela-models/TCP/props/phi1.pml
network = TEMPORARY-net-rem-9222408737716169234/8309665970987394202.pml
IO = promela-models/TCP/korg-components/IO.txt
Models?  False
ltl newPhi: (! (<> ((b==1)))) || ([] ((! ((state[0]==0))) || (! ((state[1]==4)))))
pan:1: assertion violated  !( !(( !((state[0]==0))|| !((state[1]==4))))) (at depth 23)
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml1.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml2.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml3.trail
pan:4: assertion violated  !(( !(( !((state[0]==0))|| !((state[1]==4))))&&(b==1))) (at depth 23)
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml4.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml5.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml6.trail
pan:7: assertion violated  !((b==1)) (at depth 29)
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml7.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml8.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml9.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml10.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml11.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml12.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml13.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml14.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml15.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml16.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml17.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml18.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml19.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml20.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml21.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml22.trail
pan: wrote attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml23.trail
```

Second, if a property is not supported by the extracted FSM, we print a message saying that because it is not supported, it cannot be used to synthesize candidate attackers.  For instance:

```
promela-models/TCP/props/phi3.pml was not supported by TEMPORARY-net-rem-7326678231000585354/4934667472316810343.pml || TEMPORARY-net-rem-7326678231000585354/288689041877425805.pml so cannot be attacked.
promela-models/TCP/props/phi5.pml was not supported by TEMPORARY-net-rem-7326678231000585354/4934667472316810343.pml || TEMPORARY-net-rem-7326678231000585354/288689041877425805.pml so cannot be attacked.
```

On the other hand, if a property is supported by the extracted FSM, then we use KORG to generate candidate attackers.  The step where KORG generates the candidates looks like this:

```
out/attack-promela-models.TCP.props.phi1-TCP-_True/attacker_0_WITH_RECOVERY.pml is an attack with recovery against
			promela-models/TCP/props/phi1.pml
out/attack-promela-models.TCP.props.phi1-TCP-_True/attacker_0_WITH_RECOVERY_soft_transitions.pml is an attack with recovery against
			promela-models/TCP/props/phi1.pml
out/attack-promela-models.TCP.props.phi1-TCP-_True/attacker_3_WITH_RECOVERY.pml is an attack with recovery against
			promela-models/TCP/props/phi1.pml
out/attack-promela-models.TCP.props.phi1-TCP-_True/attacker_3_WITH_RECOVERY_soft_transitions.pml is an attack with recovery against
			promela-models/TCP/props/phi1.pml
out/attack-promela-models.TCP.props.phi1-TCP-_True/attacker_13_WITH_RECOVERY.pml is an attack with recovery against
			promela-models/TCP/props/phi1.pml
out/attack-promela-models.TCP.props.phi1-TCP-_True/attacker_13_WITH_RECOVERY_soft_transitions.pml is an attack with recovery against
			promela-models/TCP/props/phi1.pml
```

Importantly, these are only candidates!  They are not actually confirmed.  In other words, all we've done is check that they work against the Promela program derived from the extracted FSM; we haven't yet checked that they also work against the Canonical version.  Next, we check if the candidates work against the corresponding Canonical Promela program.  The output from this step looks like this (and again, only the ones with `_soft_` in their names interest us!):

```
TODO
```

Ok, this is the most complicated part.  In brief:

* We are only interested in the candidates that have `_soft_` in their name.  These are the ones we have modified to support partial FSMs.  If you're curious, the relevant code can be found [here](https://github.com/anonymous-sp-submission/korg-update/blob/44d2de312c56d0b844ddff5a5b1dee7082b35ca7/korg/Construct.py#L122).

* The term "with recovery" is very misleading.  The term comes from the [original KORG paper](https://arxiv.org/pdf/2004.01220.pdf), in Definition 7.  Basically these are the attacks which succeed even when we assume that they eventually terminate.  They are the only attacks we are interested in and report on, in our paper, because we consider non-terminating attacks to be unrealistic.

* We consider a candidate to be "confirmed" if it causes the Canonical program to violate at least one of its correctness properties.

* The term "transfers" is used in our code, but not in the paper.  It just means we are checking if the attacks also work against the Canonical program, i.e., if the candidates are confirmed.