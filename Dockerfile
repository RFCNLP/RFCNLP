FROM ubuntu:latest
# 1. COPY all the source code.
RUN mkdir rfcnlp
COPY . rfcnlp/
# 2. Install other dependencies
# 2.1. Stuff in requirements.txt
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y apt-utils python3-pip
WORKDIR rfcnlp
RUN python3 -m pip install --upgrade pip
RUN pip3 install --upgrade numpy
RUN pip3 install --upgrade transitions
RUN pip3 install --upgrade networkx
RUN pip3 install --upgrade matplotlib
RUN pip3 install --upgrade cython
RUN pip3 install --upgrade deepwalk
RUN pip3 install --upgrade python-Levenshtein
RUN pip3 install --upgrade lxml
RUN pip3 install --upgrade graphviz
RUN pip3 install --upgrade tabulate
RUN pip3 install --upgrade spacy
RUN pip3 install --upgrade tqdm
RUN pip3 install --upgrade scikit-learn
RUN pip3 install --upgrade torch
RUN pip3 install --upgrade transformers
RUN pip3 install --upgrade scipy
RUN apt-get install -y build-essential   \
                       python-dev        \
                       python-setuptools \
                       git
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN git clone https://github.com/pystruct/pystruct.git
WORKDIR pystruct
RUN python3 setup.py install
WORKDIR ..
RUN pip3 install --upgrade allennlp
RUN pip3 install --upgrade nltk 
# # 2.2. Install Apache OpenNLP - https://hub.docker.com/r/casetext/opennlp/dockerfile
RUN apt-get install -y openjdk-8-jdk curl maven
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME
RUN mkdir /models
RUN apt-get install -y wget
RUN curl -o /models/en-ud-ewt-sentence.bin https://www.apache.org/dyn/closer.cgi/opennlp/models/ud-models-1.0/opennlp-en-ud-ewt-sentence-1.0-1.9.3.bin
RUN curl -o /models/en-ud-ewt-pos.bin https://www.apache.org/dyn/closer.cgi/opennlp/models/ud-models-1.0/opennlp-en-ud-ewt-pos-1.0-1.9.3.bin
RUN curl -o /models/en-ud-ewt-tokens.bin https://www.apache.org/dyn/closer.cgi/opennlp/models/ud-models-1.0/opennlp-en-ud-ewt-tokens-1.0-1.9.3.bin
RUN wget https://dlcdn.apache.org/opennlp/opennlp-1.9.3/apache-opennlp-1.9.3-bin.tar.gz
RUN tar -xvzf apache-opennlp-1.9.3-bin.tar.gz
RUN mv apache-opennlp-* /usr/bin/.
# 2.3. Install GMatch4py
RUN git clone https://github.com/Jacobe2169/GMatch4py.git
WORKDIR GMatch4py
RUN pip3 install .
WORKDIR ..
# 2.4. Install our updated version of KORG
RUN git clone https://github.com/anonymous-sp-submission/korg-update.git korg-update
WORKDIR korg-update
RUN pip3 install .
# 3. Install spin.
WORKDIR ..
RUN git clone https://github.com/nimble-code/Spin.git
WORKDIR Spin
RUN apt-get install -y bison flex
RUN make install
RUN apt-get install -y graphviz
WORKDIR ..
# 4. Run NLP code.
# TODO
# 5. Run attacker synthesis.
RUN make tcp2promela
entrypoint [""]