# RFC Parsing with NLP

This repository contains code, models, and results for our paper, which is currently in review.

The repository has been thoroughly anonymized for the double-blind review process.

## Reproduce Our Results

To use our pre-trained technical language embedding, you'll need to get them using git-LFS, like so:

```
git lfs pull
```

_You might run into an error_, because GitHub throttles LFS usage and our files are rather large.  If so, try this instead:

```
rm -rf networking_bert_rfcs_only
wget https://www.dropbox.com/s/zdr6s3erkyhsdjw/networking_bert_rfcs_only.zip?dl=0
unzip networking_bert_rfcs_only.zip?dl=0
```

Either way, you'll know you've succeeded if `networking_bert_rfcs_only/config.json` looks like this:

```
{
  "_name_or_path": "bert-base-cased",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.3.0.dev0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 28996
}
```

For easy reproducibility, we created a Dockerfile.  You can use it as follows.

First, install dependencies to allow Docker access to CUDA on your GPU(s).  For details on how to do this, refer [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).  In our case, we ran:

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update

sudo apt-get install -y nvidia-docker2

sudo systemctl restart docker
```

And tested successful installation of these dependencies via:

```
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

... which outputs a nice ASCII summary of available NVIDIA GPUs for CUDA.  Importantly, this should be *the exact same table* you get by simply running `nvidia-smi` on your host machine!  In our case, this table looked like:

```
Thu Oct 28 12:21:49 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  TITAN X (Pascal)    Off  | 00000000:83:00.0 Off |                  N/A |
| 17%   25C    P0    51W / 250W |      0MiB / 12196MiB |      3%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

Next, build the Docker image.

```
sudo docker build -t rfcnlp .
```

You may see some warnings about dependencies, but the image should build.  Example output from this step can be found [here](example.outputs/docker.build.txt).

```
sudo docker run --gpus all -it rfcnlp bash
```

You will now be able to run any of the targets in the Makefile, from a prompt giving you full access to bash inside the docker image.


### NLP Results

The NLP pipeline uses machine learning and is consequentially non-deterministic.  Hence, running the NLP targets in the Makefile can produce results that differ slightly from those reported in the paper.  We encourage you to try this and see what you get, but keep in mind that the results will depend on your CPU, GPU, and drivers.

* `make dccplineartrain` - runs our LinearCRF+R model on the DCCP RFC and saves the resulting intermediary representation to `rfcs-predicted/linear_phrases/DCCP.xml`.  Example terminal output can be found [here](example.outputs/dccplineartrain.txt).

* `make tcplineartrain` - runs our LinearCRF+R model on the TCP RFC and saves the resulting intermediary representation to `rfcs-predicted/linear_phrases/TCP.xml`.  Example terminal output can be found [here](example.outputs/tcplineartrain.txt).

* `make dccpberttrain` - runs our NeuralCRF+R model on the DCCP RFC and saves the resulting intermediary representation to `rfcs-predicted/bert_pretrained_rfcs_crf_phrases_feats/DCCP.xml`.  Example terminal output can be found [here](example.outputs/dccpberttrain.txt).

* `make tcpberttrain` - runs our NeuralCRF+R model on the TCP RFC and saves the resulting intermediary representation to `rfcs-predicted/bert_pretrained_rfcs_crf_phrases_feats/TCP.xml`.  Example terminal output can be found [here](example.outputs/tcpberttrain.txt).

### FSM Extraction & Attacker Synthesis Results 

We do FSM Extraction and Attacker Synthesis all at once.  The relevant targets are defined below.  Synthesized attacks are saved to the `out` directory, and are also analyzed in the CLI output.  Additionally, the extracted FSM is saved in `TCP.png` in the case of the TCP targets, or `DCCP.png` in the case of the DCCP targets.  The FSM is converted to a Promela program, which is saved in `TCP.pml` in the case of the TCP targets, or `DCCP.pml` in the case of the DCCP targets.  

Of course, if you are running these targets inside the Docker image, all of these output files will be inside the image, so you will need to move them to the host machine if you want to inspect them in detail.  We show how to do this later on in this README.  

* `make tcp2promela` - runs FSM Extraction and Attacker Synthesis on the GOLD TCP intermediary representation.

* `make dccp2promela` - runs FSM Extraction and Attacker Synthesis on the GOLD DCCP intermediary representation.  Example terminal output can be found [here](example.outputs/dccp2promela.txt).

The targets for FSM Extraction and Attacker Synthesis against the NLP-derived intermediary representations are given below.

* `make tcplinear2promela` - runs FSM Extraction and Attacker Synthesis on the TCP LinearCRF+R intermediary representation.

* `make dccplinear2promela` - runs FSM Extraction and Attacker Synthesis on the DCCP LinearCRF+R intermediary representation.

* `make tcpbert2promela` - runs FSM Extraction and Attacker Synthesis on the TCP NeuralCRF+R intermediary representation.

* `make dccpbert2promela` - runs FSM Extraction and Attacker Synthesis on the DCCP NeuralCRF+R intermediary representation.

The machine learning step introduces some non-determinism, so your results might differ from those reported in our paper.  But, you can reproduce our results using our saved intermediary representations, using the targets given below.

* `make tcplinearpretrained2promela` - runs FSM Extraction and Attacker Synthesis on the specific TCP LinearCRF+R intermediary representation generated on our machine and used in our paper, which is stored [here](rfcs-predicted-paper/linear_phrases/TCP.xml).

* `make dccplinearpretrained2promela` - runs FSM Extraction and Attacker Synthesis on the specific DCCP LinearCRF+R intermediary representation generated on our machine and used in our paper, which is stored [here](rfcs-predicted-paper/linear_phrases/DCCP.xml).

* `make tcpbertpretrained2promela` - runs FSM Extraction and Attacker Synthesis on the specific TCP NeuralCRF+R intermediary representation generated on our machine and used in our paper, which is stored [here](rfcs-predicted-paper/bert_pretrained_rfcs_crf_phrases_feats/TCP.xml).

* `make dccpbertpretrained2promela` - runs FSM Extraction and Attacker Synthesis on the specific DCCP NeuralCRF+R intermediary representation generated on our machine and used in our paper, which is stored [here](rfcs-predicted-paper/bert_pretrained_rfcs_crf_phrases_feats/DCCP.xml).

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

# TCP LinearCRF+R
make tcplinear2promela
mv out tcplinear2promela.out
make clean

# DCCP LinearCRF+R
make dccplinear2promela
mv out dccplinear2promela.out
make clean

# TCP NeuralCRF+R
make tcpbert2promela
mv out tcpbert2promela.out
make clean

# DCCP NeuralCRF+R
make dccpbert2promela
mv out dccpbert2promela.out
make clean
```

By the way, to exit the Docker image, you simply enter `exit`.

```
root@6abd73c26503:/rfcnlp# exit
exit
```

Then you could copy the results out of the Docker image and into your host computer for inspection.  Notice the use of `flamboyant_tu`; you'll have to replace this with your image's name.

```
docker cp flamboyant_tu:rfcnlp/tcp2promela.out/        tcp2promela.out/
docker cp flamboyant_tu:rfcnlp/dccp2promela.out/       dccp2promela.out/
docker cp flamboyant_tu:rfcnlp/tcplinear2promela.out/  tcplinear2promela.out/
docker cp flamboyant_tu:rfcnlp/dccplinear2promela.out/ dccplinear2promela.out/
docker cp flamboyant_tu:rfcnlp/dccpbert2promela.out/   dccpbert2promela.out/
```

There is a strange issue where when you do this, you don't have non-`sudo` permission to change or delete the copied files.  But you do have `sudo` permission, e.g., you can do `sudo rm -rf dccpbert2promela.out` if you'd like.

Another thing to be aware of is that we liberally color the terminal output from our scripts for readability.  So if you are piping our output into your own software, you may run into issues.  These can be easily resolved by modifying [our coloring logic](nlp2promela/printUtils.py).

## Command-Line Interface

After interacting with the predefined [Makefile](Makefile) targets in the [Dockerfile](Dockerfile), you might want to start interacting directly with our software.  In this section, we explain the specific software provided, and how to interact with it.

### `nlp2promela/nlp2promela.py`

This script takes only one argument, namely the path to the intermediate representation of an RFC document.  It performs FSM Extraction on the provided intermediate representation.  Without loss of generality, if the path to the intermediate representation was `/dir1/dir2/dir3/protocol3.xml`, then it saves the extracted FSM as an image in `protocol3.png`, and as a Promela program in `protocol3.pml`.  

In the special case where the FSM name (in the example, `protocol3`) contains the sub-string "TCP" or "DCCP", respectively, it compares the graph representation of the extracted FSM to a canonical graph representation of TCP (or DCCP, resp.), stored in [testConstants.py](nlp2promela/testConstants.py), and outputs a detailed summary of the differences.  Then, it performs Attacker Synthesis on the extracted FSM, using the TCP (or DCCP, resp.) correctness properties stored in `promela-models/TCP/props`, and saves any synthesized attackers to `out/`.

Example usage:

```
python3 nlp2promela/nlp2promela.py rfcs-predicted/bert_pretrained_rfcs_crf_phrases_feats/DCCP.xml
```

Interpreting the output in detail is complicated, because of how we use [KORG](https://github.com/anonymous-sp-submission/korg-update).  For more details, refer to the [attacker synthesis tutorial](tutorials/attacker.synthesis.md).


### `nlp-parser/linear.py`

This script uses a Linear-CRF model to learn to predict an intermediate
representation for a given protocol. It can be run by doing:

```
python3 nlp-parser.py linear.py \
	--protocol PROTOCOL         \
	[--stem]                    \
	[--write_results]           \
	[--heuristics]              \
	[--heuristics_only]         \
	[--token_level]             \
	[--phrase_level]            \
	--outdir OUTDIR
```

#### Parameters


* `--protocol`: receives one of BGPv4, DCCP, LTP, PPTP, SCTP, TCP
* `--token_level`: if passed, the script will perform token-level prediction
* `--phrase_level`: if passed, the script will perform phrase-level
  prediction
* `--stem`: if passed, the script will perform stemming of the language input (best results include stemming)
* `--heuristics`: if passed, the script will apply the post-processing
  correction rules on top of the predicted output (referred to as
LinearCRF+R in the paper)
* `--heuristics_only`: if passed, the script will not perform any
  learning, and just rely on post-processing correction rules for
prediction (referred to as Rule-based baseline in the paper)
* `--outdir`: receives an existing directory to save the resulting intermediate representation
* `--write_results`: if passed, the resulting intermediate
  representation will be written in the directory specified in `OUTDIR`

### `nlp-parser/bert_bilstm_crf.py`

This script uses a Bert-BiLSTM-CRF model to learn to predict an intermediate
representation for a given protocol. It can be run by doing:

```
python3 bert_bilstm_crf.py          \
	--protocol PROTOCOL             \
	[--features]                    \
	[--batch_size BATCH_SIZE]       \
	[--patience PATIENCE]           \
	--savedir SAVEDIR               \
	[--do_train]                    \
	[--do_eval]                     \
	--outdir OUTDIR                 \
	[--write_results]               \
	[--heuristics]                  \
	--bert_model BERT_MODEL         \
	[--learning_rate LEARNING_RATE] \
	[--cuda_device CUDA_DEVICE]
```
#### Parameters

* `--protocol`: receives one of BGPv4, DCCP, LTP, PPTP, SCTP, TCP
* `--features`: if passed, the full set of features will be extracted.
  If not passed, only BERT embeddings will be used as input
* `--batch_size`: needs to be 1 to work correctly, as the CRF layer
  does not handle batch processing successfully.
* `--patience`: receives number of epochs to wait after no improvement is
  observed in the development set for early stopping.
* `--bert_model`: receives path to the pre-trained bert model, use
  `bert-base-cased`: if you want to use standard BERT. Use `networking_bert_rfcs_only` if you want to use our technical BERT embeddings.
* `--learning_rate`: receives learning rate to use. We use `2e-5` in the
  paper.
* `--heuristics`: if passed, the script will apply the post-processing
  correction rules on top of the predicted output (referred to as
NeuralCRF+R in the paper)
* `--savedir`: receives directory to save the trained pytorch model
* `--do_train`: if passed, training will be performed. If not passed,
  model saved in `SAVEDIR` will be used as checkpoint for prediction
* `--do_eval`: if passed, prediction and evaluation will be performed
  using the model saved in `SAVEDIR`
* `--outdir`: receives an existing directory to save the resulting intermediate representation
* `--write_results`: if passed, the resulting intermediate
  representation will be written in the directory specified in `OUTDIR`
* `--cuda_device`: receives an int specifying which of your GPU devices
  to use. It defaults to 0.

## Disclaimers

Considerable effort was taken to anonymize and automate our code-base for open-source release.  If you encounter something that does not work as expected, please feel free to open a [GitHub Issue](https://github.com/anonymous-sp-submission/RFCNLP/issues) reporting the problem, and we will do our best to (anonymously) resolve it.
