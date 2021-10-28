cleanTemporary:
	- rm -rf TEMPORARY*

clean:
	- rm *.trail
	- rm *.pml
	- rm *.png
	- rm -rf out
	- rm -rf net-rem-*
	- rm _spin_nvr.tmp
	- rm *tmp*
	- rm pan*
	- rm ._n_i_p_s_
	make cleanTemporary
	- rm dot.*

.PHONY: nlp2promela

# --- Attack Synthesis Targets ---

tcp2promela:
	python3 nlp2promela/nlp2promela.py rfcs-annotated-tidied/TCP.xml
	make cleanTemporary

dccp2promela:
	python3 utils/tidy_documents.py
	python3 nlp2promela/nlp2promela.py rfcs-annotated-tidied/DCCP.xml
	make cleanTemporary

tcplinearpretrained2promela:
	python3 nlp2promela/nlp2promela.py rfcs-predicted-paper/linear_phrases/TCP.xml
	make cleanTemporary

dccplinearpretrained2promela:
	python3 nlp2promela/nlp2promela.py rfcs-predicted-paper/linear_phrases/DCCP.xml
	make cleanTemporary

tcpbertpretrained2promela:
	python3 nlp2promela/nlp2promela.py rfcs-predicted-paper/bert_pretrained_rfcs_crf_phrases_feats/TCP.xml
	make cleanTemporary

dccpbertpretrained2promela:
	python3 nlp2promela/nlp2promela.py rfcs-predicted-paper/bert_pretrained_rfcs_crf_phrases_feats/DCCP.xml
	make cleanTemporary

tcplinear2promela:
	python3 nlp2promela/nlp2promela.py rfcs-predicted/linear_phrases/TCP.xml
	make cleanTemporary

dccplinear2promela:
	python3 nlp2promela/nlp2promela.py rfcs-predicted/linear_phrases/DCCP.xml
	make cleanTemporary

tcpbert2promela:
	python3 nlp2promela/nlp2promela.py rfcs-predicted/bert_pretrained_rfcs_crf_phrases_feats/TCP.xml
	make cleanTemporary

dccpbert2promela:
	python3 nlp2promela/nlp2promela.py rfcs-predicted/bert_pretrained_rfcs_crf_phrases_feats/DCCP.xml
	make cleanTemporary

# --- NLP Targets ---

dccplineartrain:
	WANDB_MODE="dryrun" WANRB_API_KEY="dryrun"  \
		python3 nlp-parser/linear.py            \
		--protocol DCCP                         \
		--stem                                  \
		--heuristics                            \
		--phrase_level                          \
		--outdir rfcs-predicted/linear_phrases/ \
		--write_results

tcplineartrain:
	WANDB_MODE="dryrun" WANRB_API_KEY="dryrun"  \
		python3 nlp-parser/linear.py            \
		--protocol TCP                          \
		--stem                                  \
		--heuristics                            \
		--phrase_level                          \
		--outdir rfcs-predicted/linear_phrases/ \
		--write_results

dccpberttrain:
	WANDB_MODE="dryrun" WANRB_API_KEY="dryrun"                          \
		python3 nlp-parser/bert_bilstm_crf.py                           \
		--features                                                      \
		--savedir .                                                     \
		--do_train                                                      \
		--do_eval                                                       \
		--heuristics                                                    \
		--protocol DCCP                                                 \
		--outdir rfcs-predicted/bert_pretrained_rfcs_crf_phrases_feats/ \
		--write_results                                                 \
		--bert_model networking_bert_rfcs_only                          \
		--learning_rate 2e-5                                            \
		--batch_size 1

tcpberttrain:
	WANDB_MODE="dryrun" WANRB_API_KEY="dryrun"                          \
		python3 nlp-parser/bert_bilstm_crf.py                           \
		--features                                                      \
		--savedir .                                                     \
		--do_train                                                      \
		--do_eval                                                       \
		--heuristics                                                    \
		--protocol TCP                                                  \
		--outdir rfcs-predicted/bert_pretrained_rfcs_crf_phrases_feats/ \
		--write_results                                                 \
		--bert_model networking_bert_rfcs_only                          \
		--learning_rate 2e-5                                            \
		--batch_size 1
