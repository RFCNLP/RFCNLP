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

subs: ; git submodule update --init --recursive

install:
	git submodule update --recursive --remote
	sudo pip3 install GMatch4py/.
	sudo pip3 install --upgrade --force-reinstall korg-update/.

.PHONY: nlp2promela

tcp2promela:
	python3 nlp2promela/nlp2promela.py rfcs-annotated-tidied/TCP.xml
	make cleanTemporary

dccp2promela:
	python3 utils/tidy_documents.py
	python3 nlp2promela/nlp2promela.py rfcs-annotated-tidied/DCCP.xml
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