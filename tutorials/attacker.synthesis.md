# Attacker Synthesis Tutorial

This tutorial consists of the following sections.

1. **NLP Attacker Synthesis:** How to generate attacks using the intermediary representations (Gold, NeuralCRF+R, LinearCRF+R).
2. **Canonical Attacker Synthesis:** How to generate attacks using the Canonical FSMs.
3. **Confirmation of Candidate Attackers:** How to check if an attack generated using one of the intermediary representations is actually confirmed against the corresponding Canonical FSM.
4. **Attacker Strategies:** How to characterize the strategy of a confirmed attack, in a semi-automated fashion.

## 1. NLP Attacker Synthesis

The easiest way to run the NLP attacker synthesis code is to use one of the Makefile targets, as described in the [README](../README.md).  For example, suppose we want to extract an FSM from the `TCP LinearCRF+R` intermediary representation, and then synthesize attacks using this FSM and our predefined correctness properties.  Then we would run, in Bash:

```
python3 nlp2promela/nlp2promela.py rfcs-predicted-paper/linear_phrases/TCP.xml
```

Equivalently, we could simply use `make tcplinear2promela`.  The terminal output will begin with some debug information which is useful for development purposes, but not for actually using the tool.  In [our example output](../example.outputs/tcplinear2promela.txt), this debug information is printed in lines 1 through 34.  You can ignore these lines.

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

Pen-ultimately, we save files `TCP.pml` and `TCP.png`, if the protocol is TCP, or `DCCP.pml` and `DCCP.png`, if the protocol is DCCP.  Either way, these files contain the extracted FSM, in Promela and as a diagram, respectively.  For example, here is the automatically generated diagram for the TCP LinearCRF+R FSM:

![TCP Linear Diagram](../example.outputs/tcplinear2promela/TCP.png)

This image can be compared to the corresponding Tikz diagram in the Appendix of our paper (Figures 8, 9, and 10).

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
For your convenience, we include the script [checkAttack.py](checkAttack.py).  The usage is `python3 checkAttack.py [attackFile]`, e.g., `python3 checkAttack.py example.outputs/dccplinear2promela/attack-promela-models.DCCP.props.phi1-DCCP-_True/attacker_96_WITH_RECOVERY_soft_transitions.pml`.  The script will check if a candidate attacker is confirmed or not, and print a detailed explanation to the terminal.

## 4. Attacker Strategies

TODO