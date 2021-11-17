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

Our code can be used to perform attacker synthesis against other RFCs, but you need to write correctness properties first.

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

The word "supports" might seem strange -- you can interpret it as meaning "verifies", "makes true", etc.  Basically, `P⊨phi` means `phi` is a property which is true about the Promela program `P`.

These results are reported in Table IV of our paper.

Next, we synthesize condidate attackers using each of the supported properties.  These get saved to the folder `out/`.
Meanwhile, the terminal output contains output from running Spin, which is useful from a development perspective, but not very useful for the end user.

Once the attacks are synthesized and saved in the `out/` folder, only the ones with `_soft` in their file-name are relevant.  (The other ones do not incorporate our changes to support partial FSMs.)  For instance, Attacker 32 produced using `phi1` with TCP Gold:

```
/* spin -t32 -s -r -p -g attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml */
active proctype attacker() {
	
	if
	:: BtoN ? SYN;
	fi unless timeout;
	if
	:: BtoN ? ACK;
	fi unless timeout;
	if
	:: NtoB ! ACK;
	fi unless timeout;
// recovery to N
// N begins here ... 

	do
	:: AtoN ? SYN -> 
		if
		:: NtoB ! SYN;
		fi unless timeout;

	:: AtoN ? ACK -> 
		if
		:: NtoB ! ACK;
		fi unless timeout;

	:: AtoN ? FIN -> 
		if
		:: NtoB ! FIN;
		fi unless timeout;

	:: BtoN ? SYN -> 
		if
		:: NtoA ! SYN;
		fi unless timeout;

	:: BtoN ? ACK -> 
		if
		:: NtoA ! ACK;
		fi unless timeout;

	:: BtoN ? FIN -> 
		if
		:: NtoA ! FIN;
		fi unless timeout;

	od

}
```

All canonical attacks work, as KORG is sound and complete.  But an attack synthesized with an NLP-derived FSM might not always work, due to errors in the FSM.  To check if such an attack works - i.e., if it is confirmed - you can run it with the corresponding model.  For TCP, here is a nice test-harness, which will test if a TCP attack works against at least one of the TCP properties:

```
mtype = { SYN, FIN, ACK, ABORT, CLOSE, RST, OPEN }

chan AtoN = [1] of { mtype };
chan NtoA = [0] of { mtype };
chan BtoN = [1] of { mtype };
chan NtoB = [0] of { mtype };

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

bit b = 0;

proctype TCP(chan snd, rcv; int i) {
	pids[i] = _pid;
CLOSED:
	state[i] = ClosedState;
	if
	/* Passive open */
	:: goto LISTEN;
	/* Active open */
	:: snd ! SYN; goto SYN_SENT;
	/* Terminate */
	:: goto end;
	fi
LISTEN:
	state[i] = ListenState;
	if
	:: rcv ? SYN -> snd ! SYN; 
	                snd ! ACK; goto SYN_RECEIVED;
	/* Simultaneous LISTEN */
	:: timeout -> goto CLOSED; 
	/* recently added the 'timout.' */
	fi
SYN_SENT:
	state[i] = SynSentState;
	if
	:: rcv ? SYN;
		if
		/* Standard behavior */
		:: rcv ? ACK -> snd ! ACK; goto ESTABLISHED;
		/* Simultaneous open */
		:: snd ! ACK; goto SYN_RECEIVED;
		fi
	:: rcv ? ACK; rcv ? SYN -> snd ! ACK; goto ESTABLISHED;
	/* Timeout */
	:: timeout -> goto CLOSED;
	fi
SYN_RECEIVED:
	state[i] = SynRecState;
	rcv ? ACK; goto ESTABLISHED;
	/* We may want to consider putting a timeout -> CLOSED here. */
ESTABLISHED:
	state[i] = EstState;
	if
	/* Close - initiator sequence */
	:: snd ! FIN; goto FIN_WAIT_1;
	/* Close - responder sequence */
	:: rcv ? FIN -> snd ! ACK; goto CLOSE_WAIT;
	fi
FIN_WAIT_1:
	state[i] = FinW1State;
	if
	/* Simultaneous close */
	:: rcv ? FIN -> snd ! ACK; goto CLOSING;
	/* Standard close */
	:: rcv ? ACK -> goto FIN_WAIT_2;
	fi
CLOSE_WAIT:
	state[i] = CloseWaitState;
	snd ! FIN; goto LAST_ACK;
FIN_WAIT_2:
	state[i] = FinW2State;
	rcv ? FIN -> snd ! ACK; goto TIME_WAIT;
CLOSING:
	state[i] = ClosingState;
	rcv ? ACK -> goto TIME_WAIT;
LAST_ACK:
	state[i] = LastAckState;
	rcv ? ACK -> goto CLOSED;
TIME_WAIT:
	state[i] = TimeWaitState;
	goto CLOSED;
end:
	state[i] = EndState;
}

init {
	state[0] = ClosedState;
	state[1] = ClosedState;
	run TCP(AtoN, NtoA, 0);
	run TCP(BtoN, NtoB, 1);
}

ltl all_phi {
	(eventually ( b == 1 ) ) implies (
		(
			always ( leftClosed implies !rightEstablished )
		)
		&&
		(
			( (always ( eventually ( state[0] == 1 && state[1] == 2 ) ) ) 
				implies ( eventually ( state[0] == 4 ) ) )
		)
		&&
		(
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
		)
		&&
		(
			always (
				(state[0] == SynRecState)
					implies (
						eventually (
							(state[0] == EstState   || 
							 state[0] == FinW1State ||
							 state[0] == ClosedState)
						)
					)
				)
		)
}
```

Simply append the attack to the end of the file, name it e.g. `test-TCP-attack.pml`, and then run e.g.

```
spin -run -a -DNOREDUCE test-TCP-attack.pml
```

If Spin reports no violation or acceptance cycle found, then the attack is not confirmed.  If it does find a violation or an acceptance cycle, then it saves a trail file.  You can view the trail file by running, e.g.:

```
spin -t0 -s -r test-TCP-attack.pml
```

For more information refer to the [Spin docs](http://spinroot.com/spin/whatispin.html).  For DCCP the process is the same, but the harness file to test attacks is the following:

```
mtype = { DCCP_REQUEST, 
          DCCP_RESPONSE, 
          DCCP_DATA, 
          DCCP_ACK, 
          DCCP_DATAACK, 
          DCCP_CLOSEREQ, 
          DCCP_CLOSE, 
          DCCP_RESET,
          DCCP_SYNC,
          DCCP_SYNCACK };

chan AtoN = [1] of { mtype };
chan NtoA = [0] of { mtype };
chan BtoN = [1] of { mtype };
chan NtoB = [0] of { mtype };

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

bit b = 0;

proctype DCCP(chan snd, rcv; int i) {
    bool I_am_active;
CLOSED:
    I_am_active = false;
    before_state[i] = state[i];
    state[i] = ClosedState;
    if
    :: goto LISTEN; /* passive open */
    :: snd ! DCCP_REQUEST;  /* active  open */ 
       goto REQUEST; 
    fi
LISTEN:
    before_state[i] = state[i];
    state[i] = ListenState;
    if
    :: rcv ? DCCP_REQUEST -> /* rcv request  */
       snd ! DCCP_RESPONSE; /* snd response */ 
       goto RESPOND;
    :: timeout -> goto CLOSED; // need to add this to the FSM & confirm
    fi
REQUEST:
    I_am_active = true;
    before_state[i] = state[i];
    state[i] = RequestState;
    if
    :: rcv ? DCCP_RESPONSE -> /* rcv response */
       snd ! DCCP_ACK;       /* snd Ack      */
       goto PARTOPEN;
    :: rcv ? DCCP_RESET -> goto CLOSED;
    :: rcv ? DCCP_SYNC -> snd ! DCCP_RESET; goto CLOSED;
    :: timeout -> goto CLOSED; // need to add this to the FSM & confirm
    fi
RESPOND:
    I_am_active = false;
    before_state[i] = state[i];
    state[i] = RespondState;
    /* rcv Ack/DataAck */
    do
    :: rcv ? DCCP_ACK     -> goto OPEN;
    :: rcv ? DCCP_DATAACK -> goto OPEN; 
    /* It MAY also leave the RESPOND state for CLOSED after a timeout of not less
     * than 4MSL (8 minutes); when doing so, it SHOULD send a DCCP-Reset
     * with Reset Code 2, "Aborted", to clean up state at the client. */
    :: timeout -> 
        if
        :: snd ! DCCP_RESET;
        :: skip;
        fi
        goto CLOSED;
    :: snd ! DCCP_DATA; // need to add this to the FSM & confirm
    od
PARTOPEN:
    before_state[i] = state[i];
    state[i] = PartOpenState;
    do
    /* rcv packet */
    :: rcv ? DCCP_DATA;    snd ! DCCP_ACK; goto OPEN;
    :: rcv ? DCCP_DATAACK; snd ! DCCP_ACK; goto OPEN;
    /* send packet */
    :: snd ! DCCP_DATAACK;
    /* timeout for reliability */
    :: timeout -> goto CLOSED;
    /* go to OPEN because they know about me -- but do so implicitly, because
     * it would be kind of inconvenient to process a message twice in Promela.
     */
    :: rcv ? DCCP_CLOSEREQ -> snd ! DCCP_CLOSE; goto CLOSING; // duplicate OPEN logic
    :: rcv ? DCCP_CLOSE    -> snd ! DCCP_RESET; goto CLOSED;  // duplicate OPEN logic
    :: rcv ? DCCP_ACK      -> goto OPEN;
    od
OPEN:
    before_state[i] = state[i];
    state[i] = OpenState;
    do
    /* send data */
    :: snd ! DCCP_DATA;
    :: snd ! DCCP_DATAACK;
    :: rcv ? DCCP_ACK;
    :: rcv ? DCCP_DATA;
    :: rcv ? DCCP_DATAACK;
    /* server active close */
    :: I_am_active == true -> 
       snd ! DCCP_CLOSEREQ; /* snd CloseReq */
       goto CLOSEREQ;
    :: rcv ? DCCP_CLOSE -> /* rcv Close */
       snd ! DCCP_RESET; /* snd Reset */
       goto CLOSED;
    /* active close */
    :: I_am_active == true ->
       snd ! DCCP_CLOSE; /* snd Close */ 
       goto CLOSING;
    :: rcv ? DCCP_CLOSEREQ -> /* rcv CloseReq */
       snd ! DCCP_CLOSE;    /* snd Close */ 
       goto CLOSING;
    /* simply shut down the connection silently */
    :: goto CLOSED; // need to add this to the FSM & confirm 
    od
CLOSEREQ:
    before_state[i] = state[i];
    state[i] = CloseReqState;
    rcv ? DCCP_CLOSE;  /* rcv Close */
    snd ! DCCP_RESET; /* snd Reset */
    goto CLOSED;
CLOSING:
    before_state[i] = state[i];
    state[i] = ClosingState;
    if
    :: rcv ? DCCP_RESET -> /* rcv Reset */
       goto TIMEWAIT;
    // Not in the spec
    :: timeout -> goto CLOSED;
    fi
TIMEWAIT:
    before_state[i] = state[i];
    state[i] = TimeWaitState;
    skip; /* 2MSL timer expires*/
    goto CLOSED;
}

init {
    state[0] = ClosedState;
    state[1] = ClosedState;
    before_state[0] = ClosedState;
    before_state[1] = ClosedState;
    run DCCP(AtoN, NtoA, 0);
    run DCCP(BtoN, NtoB, 1);
}

ltl all_phi {
    (eventually ( b == 1 ) ) implies (
    	(
    	    ! ( eventually 
    	        always (
    	            (state[0] == before_state[0]) &&
    	            (state[1] == before_state[1])
    	        )
    	    )
        )
        &&
        (
    		always (
            	!(state[0] == TimeWaitState && state[1] == TimeWaitState)
        	)
        )
        &&
        (
        	!(eventually always (state[0] == before_state[0]))
        )
        &&
        (
        	always (
    			!(
    				state[0] == CloseReqState &&
    				state[1] == CloseReqState
    			)
    		)	
        )
    )
}
```