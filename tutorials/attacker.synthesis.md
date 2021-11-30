# Attacker Synthesis Tutorial

This tutorial consists of the following sections.

1. [**NLP Attacker Synthesis:** How to generate attacks using the intermediary representations (Gold, NeuralCRF+R, LinearCRF+R).](https://github.com/RFCNLP/RFCNLP/blob/main/tutorials/attacker.synthesis.md#1-nlp-attacker-synthesis)
2. [**Canonical Attacker Synthesis:** How to generate attacks using the Canonical FSMs.](https://github.com/RFCNLP/RFCNLP/blob/main/tutorials/attacker.synthesis.md#2-canonical-attacker-synthesis)
3. [**Confirmation of Candidate Attackers:** How to check if an attack generated using one of the intermediary representations is actually confirmed against the corresponding Canonical FSM.](https://github.com/RFCNLP/RFCNLP/blob/main/tutorials/attacker.synthesis.md#3-confirmation-of-candidate-attackers)
4. [**Attacker Strategies:** How to characterize the strategy of a confirmed attack, in a semi-automated fashion.](https://github.com/RFCNLP/RFCNLP/blob/main/tutorials/attacker.synthesis.md#4-attacker-strategies)

## 1. NLP Attacker Synthesis

The easiest way to run the NLP attacker synthesis code is to use one of the [Makefile](../Makefile) targets, as described in the [README](../README.md).  For example, suppose we want to extract an FSM from the `TCP LinearCRF+R` intermediary representation, and then synthesize attacks using this FSM and our predefined correctness properties.  Then we could run, in Bash:

```
python3 nlp2promela/nlp2promela.py rfcs-predicted-paper/linear_phrases/TCP.xml
```

Or equivalently, we could simply use:

```
make tcplinear2promela
```
The terminal output will begin with some debug information which is useful for development purposes, but not for actually using the tool.  In [our example output](../example.outputs/tcplinear2promela.txt), this debug information is printed in lines 1 through 34.  You can ignore these lines.

Second, the terminal will print the correct, incorrect, and missing states.  In [our example output](../example.outputs/tcplinear2promela.txt), this information is printed in lines 34 through 45.  Notice that `∅` is the empty set, so for example, `WRONG STATES: ∅` means that none of the extracted states were incorrect (when compared to the Canonical FSM).

Third, the terminal will print the correct communication, user call, and reset transitions.  In [our example output](../example.outputs/tcplinear2promela.txt), this information is printed in lines 46 through 83.  This information corresponds to the "Correct" column in Table III in our paper.

Fourth, the terminal will print the incorrect transitions in a table.  The columns of this table are `Source`, `Label`, `Destination`, `Line #s`, `Diagnosis`, and `Passes Filter?`.  The meanings of these entries are as follows.
* `Source`: In what state does the transition begin?
* `Label`: What events occur during the transition, if any?  For example, `FIN?;ACK!;ACK?` means "receive a `FIN`, then send an `ACK`, then receive an `ACK`".  Note that `ε` means nothing occurs at all.
* `Destination`: What state does the transition end in?
* `Line #s`: This will look something like `[519] ---> [617]`.  The first number in `[`s is the line number in the cleaned XML file, e.g., `rfcs-annotated-tidied/TCP-clean.xml`.  This is exactly like the original intermediary representation except that it has been stripped of superfluous white-space and certain special characters.  The second number in `[`s is the line number of the same quote in the original intermediary representation, which in the case of TCP LinearCRF+R, is [this file](rfcs-annotated-tidied/TCP.xml).  This information is critical for *explaining* mistakes in the FSM extraction process, because you can look at the specific information from which an incorrect transition was extracted, and figure out *why* the code made a mistake.
* `Diagnosis`: We wrote some code to try and automatically guess an explanation for what's wrong with a given transition.  It's not always right, but it's definitely useful for debugging.  This information can help you interpret the quote found at `Line #s` of the intermediary representation.
* `Passes Filter?`: We may have a simple filter for transitions.  For instance in the case of TCP, we only allow communication transitions, as opposed to e.g. user call ones.
In [our example output](../example.outputs/tcplinear2promela.txt), this debug information is printed in lines 84 through 184.  The "Partially Correct" and "Incorrect" columns of Table III in our paper are manually derived from this information.

Fifth, the terminal will print the missing transitions in a table.  Generally we only care about the communication ones.  These should be closely compared to the incorrect transitions table, since many of the missing ones are found "partially" (with some mistake in label, or in source or destination state).  In [our example output](../example.outputs/tcplinear2promela.txt), this debug information is printed in lines 185 through 412, with some Spin output mixed in.  The "Not Found" column in Table III in our paper corresponds to this information.

Sixth, we print some summary statistics of items one through five.  In [our example output](../example.outputs/tcplinear2promela.txt), this debug information is printed in lines 413 through 427.

Seventh, we check which properties are true about (or "supported by") the extracted FSM.  If the FSM `P` supports the property `phi` then we write `P ⊨ phi`; else we write `P ⊭ phi`.  In the raw text file this part can look a little strange because of how we color the output (e.g. `[93m⊭[0` instead of `⊭`), but in the actual terminal it should be clear.  In [our example output](../example.outputs/tcplinear2promela.txt), this debug information is printed in lines 428 through 435.  This information is presented in Table IV in our paper.  Some more detailed failure analysis examples are given in Table VII, in the Appendix.

Eighth, for each supported property, we attempt to synthesize attacks.  The terminal output can look like this:

```
============ TRY TO ATTACK promela-models/TCP/props/phi1.pml============
remainder = TEMPORARY-net-rem-8310298848694960302/2360061631813030910.pml
prop = promela-models/TCP/props/phi1.pml
network = TEMPORARY-net-rem-8310298848694960302/6144110669591631791.pml
IO = promela-models/TCP/korg-components/IO.txt
Models?  False

----> Calling parseAllTrails()


-------------- PARSING TRAIL:
```

... followed by a large amount of Promela output, which is useful for debugging, and then some information about the removal of duplicate files.  Or, if no attacks can be found, the output may look like this:

```
============ TRY TO ATTACK promela-models/TCP/props/phi2.pml============
remainder = TEMPORARY-net-rem-8310298848694960302/2360061631813030910.pml
prop = promela-models/TCP/props/phi2.pml
network = TEMPORARY-net-rem-8310298848694960302/6144110669591631791.pml
IO = promela-models/TCP/korg-components/IO.txt
Models?  True
We could not find any with_recovery(model, (N), phi)-attacker A.
```

Regardless, the rest of the terminal output consists of such information.  

Pen-ultimately, we save files `TCP.pml` and `TCP.png`, if the protocol is TCP, or `DCCP.pml` and `DCCP.png`, if the protocol is DCCP.  Either way, these files contain the extracted FSM, in Promela and as a diagram, respectively.  The Promela program is the result we report on in the paper.  At this time, the diagram rendering logic is imperfect, so the `.png` file might have some mistakes -- please refer to the `.pml` file instead.  If you would like to create FSM diagrams to compare with those we give in the Appendix of our paper, you can do so using [iSpin](http://spinroot.com/spin/Man/3_SpinGUI.html) (for more instructions refer [here](https://philenius.github.io/software%20quality/2020/04/09/installing-spin-on-ubuntu-19.html)).  Our diagrams in the Appendix of our paper are manually crafted in Tikz, based on the Promela code.

Finally, we save synthesized attacker programs in `out/`.  The folder names, e.g. `attack-promela-models.DCCP.props.phi3-DCCP-_True`, indicate the protocol (e.g. `DCCP`) and the property which KORG used to synthesize the attackers in that folder (e.g. `phi3`).  Within each folder, there will be an equal number of Promela files with `soft_transitions` in their name, e.g., `attacker_92_WITH_RECOVERY_soft_transitions.pml`, and without, e.g. `attacker_91_WITH_RECOVERY.pml`.  The numbers in the filenames might not match, but there is a one-to-one correspondence between these files.  Those with `soft_transitions` in their file-name are the attackers we report in the paper, which are modified to time-out rather than block trying to send or receive.  For example, using TCP LinearCRF+R and `phi1`, we synthesize [this attacker](example.outputs/tcplinear2promela/attack-promela-models.TCP.props.phi1-TCP-_True/attacker_32_WITH_RECOVERY_soft_transitions.pml):

```
/* spin -t32 -s -r -p -g attack-promela-models.TCP.props.phi1-TCP-_daisy_check.pml */
active proctype attacker() {
	
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

This attacker program consists of:
- for debug purposes, a line recording the invocation of Spin which was used to generate the violating execution from which this specific attacker was synthesized;
- the attack logic, which in this example consists of attempting to send a single `ACK` message to Peer B; then
- the original network logic, which is run when the attack terminates.
For more details refer to the [KORG paper](https://arxiv.org/abs/2004.01220) and [documentation](https://mxvh.pl/AttackerSynthesis/).  We report on synthesized attackers in Table V of our paper.

## 2. Canonical Attacker Synthesis

The easiest way to run the Canonical attacker synthesis is to use the [Makefile](https://github.com/RFCNLP/RFCNLP-korg/blob/master/Makefile) in [our modified version of KORG](https://github.com/RFCNLP/RFCNLP-korg).  
- To generate attacks against Canonical TCP, run `make tcp`.  
- To generate attacks against Canonical DCCP, run `make dccp`.  
The Makefile has various other interesting targets which you may want to run in order to understand the code in great detail, but `tcp` and `dccp` are the only important ones for reproducing our results.

As a concrete example, let's do attacker synthesis using Canonical TCP.  We navigate to the KORG source directory and type `make tcp`.  First we see the invocation of KORG which is executed by the `tcp` target in the Makefile.

```
for j in 1 2 3 4; do               \
	python3 korg/Korg.py           \
		--model=demo/TCP/TCP.pml   \
		--phi=demo/TCP/phi$j.pml   \
		--Q=demo/TCP/network.pml   \
		--IO=demo/TCP/IO.txt       \
		--max_attacks=100          \
		--with_recovery=True       \
		--name=TCP.$j              \
		--characterize=False;      \
done;
```

Next we see many lines of Spin-related output.  This content is useful for debugging.  Finally, just like with the NLP targets, attackers are saved to the folder `out/` and redundant attackers are removed.  The naming convention for TCP Canonical attacker folders is `TCP.x_True` where `x` is the number of the property that was used to find the attacks.  The naming convention for DCCP is the same, but `DCCP.x_True` instead of `TCP.x_True`.

## 3. Confirmation of Candidate Attackers

We call a candidate attacker *confirmed* if it has a terminating execution in which it violates one of the properties.
For your convenience, we include the script [checkAttack.py](checkAttack.py).  The usage is `python3 checkAttack.py [attackFile]`, e.g., `python3 checkAttack.py example.outputs/dccplinear2promela/attack-promela-models.DCCP.props.phi1-DCCP-_True/attacker_96_WITH_RECOVERY_soft_transitions.pml`.  The script will check if a candidate attacker is confirmed or not, print a detailed explanation to the terminal, and then create and save a heuristic description of how the attack works against each property.

For instance, suppose I run:

```
python3 checkAttack.py ../example.outputs/tcpbert2promela/attack-promela-models.TCP.props.phi1-TCP-_True/attacker_32_WITH_RECOVERY_soft_transitions.pml
```

First, the script determines that the attack targets TCP, as it has "TCP" in the file-name.

```
Testing against TCP.
```

Next, the script begins testing to see if the attack is confirmed against [TCP property 1](../promela-models/TCP/props/phi1.pml).  It does this by appending the attacker to the [TCP property 1 harness file](tcp.harness.files/harness1.pml), adding the line `b = 1;` inside the attacker to signal attack termination, and then model-checking the file using [Spin](http://spinroot.com/spin/whatispin.html).

```
HARNESS =  src/tcp.harness.files/harness1.pml
pan:1: assertion violated  !( !(( !((state[0]==0))|| !((state[1]==4))))) (at depth 228)
pan: wrote tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml.trail

(Spin Version 6.5.1 -- 31 July 2020)
Warning: Search not completed

Full statespace search for:
	never claim         	+ (all_phi)
	assertion violations	+ (if within scope of claim)
	acceptance   cycles 	+ (fairness disabled)
	invalid end states	- (disabled by never claim)

State-vector 108 byte, depth reached 473, errors: 1
     1158 states, stored
      906 states, matched
     2064 transitions (= stored+matched)
        0 atomic steps
hash conflicts:         0 (resolved)

Stats on memory usage (in Megabytes):
    0.150	equivalent memory usage for states (stored*(State-vector + overhead))
    0.272	actual memory usage for states
  128.000	memory used for hash table (-w24)
    0.611	memory used for DFS stack (-m10000)
  128.806	total actual memory usage



pan: elapsed time 0 seconds
ltl all_phi: (! (<> ((b==1)))) || ([] ((! ((state[0]==0))) || (! ((state[1]==4)))))

```

The warning (`Search not completed`) does not matter, since the assertion was violated.
But if no violation was found, and Spin gave this warning, we would want to increase the search depth and try again.
Having found a violating execution, the script replays it.

```
-----------------------------------
ltl all_phi: (! (<> ((b==1)))) || ([] ((! ((state[0]==0))) || (! ((state[1]==4)))))
starting claim 3
Never claim moves to line 6	[(1)]
  2:	proc  0 (:init::1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:104 (state 1)	[state[0] = 0]
  4:	proc  0 (:init::1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:105 (state 2)	[state[1] = 0]
Starting TCP with pid 3
  6:	proc  0 (:init::1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:106 (state 3)	[(run TCP(AtoN,NtoA,0))]
		queue 1 (AtoN): 
  8:	proc  2 (TCP:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:32 (state 1)	[pids[i] = _pid]
		queue 1 (AtoN): 
		pids[0] = 2
		pids[1] = 0
 10:	proc  2 (TCP:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:34 (state 2)	[state[i] = 0]
		queue 1 (AtoN): 
 12:	proc  2 (TCP:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:37 (state 3)	[goto LISTEN]
		queue 1 (AtoN): 
 14:	proc  2 (TCP:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:44 (state 9)	[state[i] = 1]
		queue 1 (AtoN): 
		state[0] = 1
		state[1] = 0
Starting TCP with pid 4
 16:	proc  0 (:init::1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:107 (state 4)	[(run TCP(BtoN,NtoB,1))]
		queue 1 (AtoN): 
		queue 3 (BtoN): 
 18:	proc  3 (TCP:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:32 (state 1)	[pids[i] = _pid]
		queue 1 (AtoN): 
		queue 3 (BtoN): 
		pids[0] = 2
		pids[1] = 3
 20:	proc  3 (TCP:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:34 (state 2)	[state[i] = 0]
		queue 1 (AtoN): 
		queue 3 (BtoN): 
 22:	proc  3 (TCP:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:37 (state 3)	[goto LISTEN]
		queue 1 (AtoN): 
		queue 3 (BtoN): 
 24:	proc  3 (TCP:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:44 (state 9)	[state[i] = 1]
		queue 1 (AtoN): 
		queue 3 (BtoN): 
		state[0] = 1
		state[1] = 1
 26:	proc  2 (TCP:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:49 (state 14)	[(timeout)]
		queue 1 (AtoN): 
		queue 3 (BtoN): 
 28:	proc  2 (TCP:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:34 (state 2)	[state[i] = 0]
		queue 1 (AtoN): 
		queue 3 (BtoN): 
		state[0] = 0
		state[1] = 1
 30:	proc  2 (TCP:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:39 Send SYN	-> queue 1 (snd)
 30:	proc  2 (TCP:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:39 (state 4)	[snd!SYN]
		queue 1 (AtoN): [SYN]
		queue 3 (BtoN): 
 32:	proc  2 (TCP:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:52 (state 18)	[state[i] = 2]
		queue 1 (AtoN): [SYN]
		queue 3 (BtoN): 
		state[0] = 2
		state[1] = 1
 34:	proc  3 (TCP:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:49 (state 14)	[(timeout)]
		queue 1 (AtoN): [SYN]
		queue 3 (BtoN): 
 36:	proc  3 (TCP:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:34 (state 2)	[state[i] = 0]
		queue 1 (AtoN): [SYN]
		queue 3 (BtoN): 
		state[0] = 2
		state[1] = 0
 38:	proc  3 (TCP:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:39 Send SYN	-> queue 3 (snd)
 38:	proc  3 (TCP:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:39 (state 4)	[snd!SYN]
		queue 1 (AtoN): [SYN]
		queue 3 (BtoN): [SYN]
 40:	proc  3 (TCP:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:52 (state 18)	[state[i] = 2]
		queue 1 (AtoN): [SYN]
		queue 3 (BtoN): [SYN]
		state[0] = 2
		state[1] = 2
 42:	proc  1 (attacker:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:115 Sent ACK	-> queue 4 (NtoB)
 42:	proc  1 (attacker:1) tmp.6e3a690f-fa91-4aad-a579-43cbae1b2d08.pml:115 (state 1)	[NtoB!ACK]
```
... etc.

This execution is then filtered to just the peer state updates, peer send and receive events, and attack termination event.
The filtered execution, or "strategy", is printed to the terminal, like so:

```
-----------------------------------
peerA in Listen
peerB in Closed
peerA in Listen
peerB in Listen
peerA in Closed
```
... etc. etc. etc. ...
```
peerA in Closed
peerB in Established
peerA in Closed
peerB in Established
Saved strategy to strategies.TCP.bert.phi1.attacker_32.phi1.strategy
```

This strategy is, as the terminal stated, saved to the ASCII text file _strategies.TCP.bert.phi1.attacker_32.phi1.strategy_.
Since the attack was confirmed against the property, the strategy file contains first a header line, and then the body of the strategy:

```
rfc-nlp-anon/tutorials$ cat strategies.TCP.bert.phi1.attacker_32.phi1.strategy
Strategy for: ../example.outputs/tcpbert2promela/attack-promela-models.TCP.props.phi1-TCP-_True/attacker_32_WITH_RECOVERY_soft_transitions.pml
peerA in Listen
peerB in Closed
peerA in Listen
peerB in Listen
peerA in Closed
peerB in Listen
peerA sends SYN
peerA in Syn-Sent
peerB in Listen
peerA in Syn-Sent
peerB in Closed
peerB sends SYN
peerA in Syn-Sent
peerB in Syn-Sent
peerB receives ACK
peerB receives SYN
peerB sends ACK
peerA in Syn-Sent
peerB in Established
peerA receives SYN
peerA sends ACK
peerA in Syn-Received
peerB in Established
peerB sends FIN
peerA in Syn-Received
peerB in Fin-Wait-1
peerA receives ACK
peerA in Established
peerB in Fin-Wait-1
peerA sends FIN
peerA in Fin-Wait-1
peerB in Fin-Wait-1
peerB receives ACK
peerA in Fin-Wait-1
peerB in Fin-Wait-2
peerB receives FIN
peerB sends ACK
peerA in Fin-Wait-1
peerB in Time-Wait
peerA in Fin-Wait-1
peerB in Closed
peerA in Fin-Wait-1
peerB in Listen
peerA receives FIN
peerA sends ACK
peerA in Closing
peerB in Listen
peerA receives ACK
peerA in Time-Wait
peerB in Listen
peerA in Closed
peerB in Listen
peerA in Listen
peerB in Listen
peerA in Listen
peerB in Closed
peerB sends SYN
peerA in Listen
peerB in Syn-Sent
peerB receives ACK
peerA receives SYN
peerA sends SYN
peerA sends ACK
peerA in Syn-Received
peerB in Syn-Sent
peerB receives SYN
peerB sends ACK
peerA receives ACK
peerA in Syn-Received
peerB in Established
peerB sends FIN
peerA in Syn-Received
peerB in Fin-Wait-1
peerB receives ACK
peerA in Syn-Received
peerB in Fin-Wait-2
peerA in Established
peerB in Fin-Wait-2
peerA sends FIN
peerA in Fin-Wait-1
peerB in Fin-Wait-2
peerA receives FIN
peerA sends ACK
peerA in Closing
peerB in Fin-Wait-2
peerB receives FIN
peerB sends ACK
peerA in Closing
peerB in Time-Wait
peerA in Closing
peerB in Closed
peerB sends SYN
peerA in Closing
peerB in Syn-Sent
peerA receives ACK
peerA in Time-Wait
peerB in Syn-Sent
peerA in Closed
peerB in Syn-Sent
peerA sends SYN
peerB receives ACK
peerB receives SYN
peerB sends ACK
peerA in Closed
peerB in Established
peerA in Closed
peerB in Established
```

On the other hand, if the attack is *not* confirmed against the property, then the saved strategy file consists of only the initial header line.  For instance, `../example.outputs/tcpbert2promela/attack-promela-models.TCP.props.phi1-TCP-_True/attacker_32_WITH_RECOVERY_soft_transitions.pml` is not confirmed against `phi2`:

```
rfc-nlp-anon/tutorials$ cat strategies.TCP.bert.phi1.attacker_32.phi2.strategy 
Strategy for: ../example.outputs/tcpbert2promela/attack-promela-models.TCP.props.phi1-TCP-_True/attacker_32_WITH_RECOVERY_soft_transitions.pml
rfc-nlp-anon/tutorials$
```

## 4. Attacker Strategies

Now that we have these automatically generated strategy files, how can we go about actually characterizing the overarching strategy of an attack?  Unfortunately, it is not (yet) straightforward.  First, an attack might exhibit very different behaviors in violating executions against one property versus violating executions against another, unrelated property.  So, the idea that an attack exhibits a single "strategy" is, in some sense, false.  Second, the strategy files only record what effect the attack had on the peers - not what precisely the attacker *did*.  To know this, you need to look at the attacker code itself.

Let's take a motivating example.  Consider [this attack](https://github.com/RFCNLP/RFCNLP-korg/blob/master/example.attacks/tcp/TCP.2_True/attacker_8_WITH_RECOVERY_soft_transitions.pml), which was generated using TCP Canonical and TCP property 2.  The attack attempts to do the following, before terminating:

1. Send a `SYN` to Peer B.
2. Receive a `SYN` from Peer B.
3. Receive an `ACK` from Peer B.
4. Send an `ACK` to Peer B.
5. Send a `FIN` to Peer B.
6. Receive an `ACK` from Peer B.
7. Send an `ACK` from Peer B.

Looking at this sequence of events, my first guess is that the attack is spoofing an active Peer B.  IE, when it does step 1, Peer B believes that the attacker is actually Peer A, having just transitioned into `SYN_SENT`.  How can I confirm this hypothesis?  Well, if we compute and then inspect the strategy files, we observe the following.

1. Against [phi1](../promela-models/TCP/props/phi1.pml), the attack leads Peer B to `FIN-WAIT-1` after Peer A has terminated.
2. Against [phi2](../promela-models/TCP/props/phi2.pml), the attack leads Peer B to get stuck in `SYN-SENT`.
3. Against [phi3](../promela-models/TCP/props/phi3.pml), the attack leads the peers to deadlock in `SYN-SENT x CLOSING`.
4. Against [phi4](../promela-models/TCP/props/phi4.pml), the attack leads Peer A to get stuck in `SYN-RECEIVED`.

Combining these facts, it's reasonably to conclude the following: This attack was confirmed against all four properties; and in all four cases, it actively spoofed Peer A in order to either lead one or both peers into a deadlock or stuck state.  This is what we'd call the "overarching strategy" of an attack.  Basically, it's an English-language expert description of what the attack does and how, at a high level, which allows us to compare similar attacks even if they have different code.