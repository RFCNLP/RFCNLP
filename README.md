# RFC Parsing with NLP

This repository contains code, models, and results for the paper *Automated Attack Synthesis by Extracting Finite State Machines from Protocol Specification Documents*, currently in review.

The repository has been thoroughly anonymized for the double-blind review process.

## Reproduce Our Results

You can either use our pre-trained NeuralCRF models, or train your own.  (More on this later.)  If you intend to use ours, you'll need to get them using git-LFS, like so:

```
git lfs pull
```

For easy reproducibility, we created a Dockerfile.  You can use it as follows.

First, build the Docker image.

```
sudo docker build -t rfcnlp .
```

You may see some warnings about dependencies, but the image should build (tested on Linux Mint with an Intel i7).

Next, run and enter the Docker image.

```
docker run rfcnlp
docker exec -it rfcnlp bash
```

You will now be able to run any of the targets in the Makefile.

### NLP Results

The NLP pipeline uses machine learning and is consequentially non-deterministic.  Hence, running the NLP targets in the Makefile can produce results that differ slightly from those reported in the paper.  We encourage you to try this and see what you get, but keep in mind that the results will depend on your CPU, GPU, and drivers.

* `make dccplineartrain` - runs our LinearCRF model on the DCCP RFC and saves the resulting intermediary representation.
* `make tcplineartrain` - runs our LinearCRF model on the TCP RFC and saves the resulting intermediary representation.
* `make dccpberttrain` - runs our NeuralCRF model on the DCCP RFC and saves the resulting intermediary representation.
* `make tcpberttrain` - runs our NeuralCRF model on the TCP RFC and saves the resulting intermediary representation.

### FSM Extraction & Attacker Synthesis Results 

We do FSM Extraction and Attacker Synthesis all at once.  The relevant targets are defined below.  Synthesized attacks are saved to the `out` directory in the Docker image, and are also listed in the CLI output.  

* `make tcp2promela` - runs FSM Extraction and Attacker Synthesis on the GOLD TCP intermediary representation.
* `make dccp2promela` - runs FSM Extraction and Attacker Synthesis on the GOLD DCCP intermediary representation.

By default, each target runs on the corresponding intermediary representation from our paper.  However, if you run the NLP Makefile targets first, you will save your own intermediary representation (which could differ from ours due to nondeterminism in the machine learning step), and the below Makefile targets will run on your version instead.  If you over-write our intermediary representation(s) but want to use them again without re-building your Dockerfile, you can find them in `rfcs-predicted-paper/`.

### Troubleshooting

The FSM Extraction & Attacker Synthesis targets will not work correctly if files from a prior run are left in the working directory.  The solution to this is to use our `make clean` target to clean up the working directory in-between running FSM Extraction & Attacker Synthesis targets.  Since `make clean` deletes the synthesized attacks from `out`, you might want to save those attacks somewhere else.  Here is an example workflow:

```
# TCP GOLD
make tcp2promela
mv out tcp2promela.out
make clean

# DCCP GOLD
make dccp2promela
mv out dccp2promela.out
make clean

# TCP LinearCRF
make tcplinear2promela
mv out tcplinear2promela.out
make clean

# DCCP LinearCRF
make dccplinear2promela
mv out dccplinear2promela.out
make clean

# TCP NeuralCRF
make tcpbert2promela
mv out tcpbert2promela.out
make clean

# DCCP NeuralCRF
make dccpbert2promela
mv out dccpbert2promela.out
make clean
```

Then in another terminal, you could copy the results to host for inspection:

```
docker cp rfcnlp:rfcnlp/tcp2promela.out/ tcp2promela.out/
docker cp rfcnlp:rfcnlp/dccp2promela.out/ dccp2promela.out/
docker cp rfcnlp:rfcnlp/tcplinear2promela.out/ tcplinear2promela.out/
docker cp rfcnlp:rfcnlp/dccplinear2promela.out/ dccplinear2promela.out/
docker cp rfcnlp:rfcnlp/dccpbert2promela.out/ dccpbert2promela.out/
```
