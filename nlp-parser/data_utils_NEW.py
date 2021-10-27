import nltk
from nltk.stem import PorterStemmer
from allennlp.predictors.predictor import Predictor
import re
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm, trange
import spacy
from transformers import AutoTokenizer

nlp = spacy.load("en_core_web_sm")

class ChunkDataset(data.Dataset):
    def __init__(self, x, x_feats, x_len, x_chunk_len, labels):
        self.x = torch.from_numpy(x).long()
        self.x_feats = torch.from_numpy(x_feats).float()
        self.x_len = torch.from_numpy(x_len).long()
        self.x_chunk_len = torch.from_numpy(x_chunk_len).long()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x_i = self.x[index]
        x_feats_i = self.x_feats[index]
        x_len_i = self.x_len[index]
        label = self.labels[index]
        x_chunk_len_i = self.x_chunk_len[index]
        return x_i, x_feats_i, x_len_i, x_chunk_len_i, label


def get_definitions(variables=set(), states=set(), events=set(), events_expand=set()):
    with open("rfcs-definitions/def_vars.txt") as fp:
        for line in fp:
            args = line.strip().split('\t')
            #if protocol not in variables:
            #    variables[protocols] = set()
            #variables[protocols].add(var)
            variables.add(args[1])
    with open("rfcs-definitions/def_states.txt") as fp:
        for line in fp:
            args = line.strip().split('\t')
            states.add(args[1])
    with open("rfcs-definitions/def_events.txt") as fp:
        for line in fp:
            args = line.strip().split('\t')
            events.add(args[1])
            events_expand.add(args[1])
            # Ad-hoc things for our protocols, make it more adjustable to any protocol
            if args[1].startswith('DCCP-'):
                events_expand.add(args[1][5:])

def get_protocol_definitions(protocol, states={}, events={}, events_expanded={}, variables={}):
    with open("rfcs-definitions/def_states.txt") as fp:
        for line in fp:
            args = line.strip().split('\t')
            #print(args)
            if args[0] == protocol:
                states[args[1].lower()] = args[2]
    with open("rfcs-definitions/def_vars.txt") as fp:
        for line in fp:
            args = line.strip().split('\t')
            #print(args)
            if args[0] == protocol:
                variables[args[1].lower()] = args[1].lower()
    with open("rfcs-definitions/def_events.txt") as fp:
        for line in fp:
            args = line.strip().split('\t')
            if args[0] == protocol:
                events[args[1].lower()] = args[2]
                events_expanded[args[1].lower()] = args[2]
                if args[1].startswith('DCCP-'):
                    events_expanded[args[1][5:].lower()] = args[2]

        # Add other lexical forms for specific events
        if protocol == "TCP":
            '''
            keys = list(events_expanded.keys())
            for key in keys:
                if key not in ["syn", "fin", "ack", "syn_ack"]:
                    if key in events:
                        events.pop(key)
                    if key in events_expanded:
                        events_expanded.pop(key)
            '''
            if 'ack' in events:
                events['acknowledgment'] = events['ack']
                events_expanded['acknowledgment'] = events_expanded['ack']
            if 'rst' in events:
                events['reset'] = events['rst']
                events_expanded['reset'] = events_expanded['rst']

            #print(events.keys())
            #print(events_expanded.keys())
            #exit()

def flatten(y):
    y_new = []
    for seq in y:
        y_new += list(seq)
    return y_new

def translate(y, id2tag):
    y_new = []
    for seq in y:
        y_new.append([id2tag[i] for i in seq])
        #print([id2tag[i] for i in seq])
    return y_new

def expand(X, y, id2word, debug=False):
    X_new = []; y_new = []
    for seq, y_seq in zip(X, y):
        y_new_seq = []; x_new_seq = []
        for chunk, y_chunk in zip(seq, y_seq):
            x_new_seq += chunk[0]
            if y_chunk.startswith('B') and len(chunk[0]) > 1:
                inside = re.sub('B-', 'I-', y_chunk)
                y_new_seq += [y_chunk] + [inside] * (len(chunk[0])-1)
            elif len(chunk) > 1:
                y_new_seq += [y_chunk] * (len(chunk[0]))
            else:
                y_new_seq += [y_chunk]
        #print(len(x_new_seq), len(y_new_seq))
        #exit()
        X_new.append(x_new_seq)
        y_new.append(y_new_seq)
    return X_new, y_new

def alternative_expand(X, y, level_h, level_d, id2word, debug=False):
    X_new = []; y_new = []; level_h_new = []; level_d_new = []
    for seq, y_seq, h_seq, d_seq in zip(X, y, level_h, level_d):
        y_new_seq = []; x_new_seq = []; h_new_seq = []; d_new_seq = []
        #print(y_seq)
        for chunk, y_chunk, h_chunk, d_chunk in zip(seq, y_seq, h_seq, d_seq):
            x_new_seq.append(chunk[0])
            h_new_seq.append([h_chunk] * len(chunk[0]))
            d_new_seq.append([d_chunk] * len(chunk[0]))

            if y_chunk.startswith('B') and len(chunk[0]) > 1:
                inside = re.sub('B-', 'I-', y_chunk)
                y_new_seq.append([y_chunk] + [inside] * (len(chunk[0])-1))
            elif len(chunk) > 1:
                y_new_seq.append([y_chunk] * (len(chunk[0])))
            else:
                y_new_seq.append([y_chunk])
        X_new.append(x_new_seq)
        y_new.append(y_new_seq)
        level_h_new.append(h_new_seq)
        level_d_new.append(d_new_seq)
    return X_new, y_new, level_h_new, level_d_new

def alternative_join(X, y, level_h, level_d, id2word, debug=False):
    X_new = []; y_new = []; level_h_new = []; level_d_new = []
    for seq, y_seq, h_seq, d_seq in zip(X, y, level_h, level_d):
        y_new_seq = []; x_new_seq = []; h_new_seq = []; d_new_seq = []
        for chunk, y_chunk, h_chunk, d_chunk in zip(seq, y_seq, h_seq, d_seq):
            x_new_seq += chunk
            y_new_seq += y_chunk
            h_new_seq += h_chunk
            d_new_seq += d_chunk
        X_new.append(x_new_seq)
        y_new.append(y_new_seq)
        level_h_new.append(h_new_seq)
        level_d_new.append(d_new_seq)
    return X_new, y_new, level_h_new, level_d_new

def get_token(elems, stem, word2id, next_id, pos2id, pos_id, id2cap, stem2id, next_stem_id, id2word):
    if elems[0] not in word2id:
        word2id[elems[0]] = next_id
        id2word[next_id] = elems[0]
        next_id += 1
    if elems[0].lower() not in word2id:
        word2id[elems[0].lower()] = next_id
        id2word[next_id] = elems[0].lower()
        next_id += 1
    if elems[2] not in pos2id:
        pos2id[elems[2]] = pos_id
        pos_id += 1

    if stem not in stem2id:
        stem2id[stem] = next_stem_id
        next_stem_id += 1

    # 0 for lower or capitalized, 1 for all caps, 2 for capitalized variable, 3 for camel case, 4 other form of apha, 5 numbers, 6 symbols
    if elems[0].islower() or re.match(r'^[A-Z]{1}[a-z_-]+$', elems[0]):
        id2cap[word2id[elems[0]]] = 0
    elif elems[0].isupper():
        id2cap[word2id[elems[0]]] = 1
    elif re.match(r'^([A-Z]{1}[a-z_-]+)+$', elems[0]):
        id2cap[word2id[elems[0]]] = 2
    elif re.match(r'^[a-z]+([A-Z][a-z_-]+)+$', elems[0]):
        id2cap[word2id[elems[0]]] = 387
    elif re.match(r'[a-zA-Z]+[a-zA-Z_-]+', elems[0]):
        id2cap[word2id[elems[0]]] = 4
    elif re.match(r'[0-9]+',  elems[0]):
        id2cap[word2id[elems[0]]] = 5
    else:
        id2cap[word2id[elems[0]]] = 6
    return next_id, pos_id, next_stem_id

def get_data_nochunks(files, word2id={}, tag2id={}, pos2id={}, id2cap={}, stem2id={}, id2word={}):
    ps = PorterStemmer()
    next_id = len(word2id)
    if next_id == 0:
        word2id["[PAD]"] = 0
        next_id = 1
    next_tag_id = len(tag2id)
    next_stem_id = len(stem2id)
    pos_id = len(pos2id)
    X_train = []; y_train = []
    #x_chunk = [[],[],[]];
    x_control = [[],[],[]]; y_control = []

    for f in files:
        with open(f) as fp:
            for line in fp:
                elems = line.split(' =======')
                if elems[0] == "END-OF-CHUNK":
                    continue
                elif elems[0] == "END-OF-CONTROL":
                    y_train.append(np.array(y_control))
                    X_train.append(x_control)
                    y_control = []; x_control = [[],[],[]]
                else:
                    stem = ps.stem(elems[0].lower())
                    next_id, pos_id, next_stem_id = get_token(elems, stem, word2id, next_id, \
                                                               pos2id, pos_id, id2cap, \
                                                               stem2id, next_stem_id, id2word)

                    x_control[0].append(word2id[elems[0]])
                    x_control[1].append(pos2id[elems[2]])
                    x_control[2].append(stem2id[stem])
                    
                    # TO-DO: Add a warning here
                    if elems[1].strip().startswith('B-REF') or elems[1].strip().startswith('I-REF'):
                        curr_tag = 'O'
                    else:
                        curr_tag = elems[1].strip()

                    if curr_tag not in tag2id:
                        tag2id[curr_tag] = next_tag_id
                        next_tag_id += 1


                    y_control.append(tag2id[curr_tag])

    return X_train, np.array(y_train)

def get_data(files, word2id={}, tag2id={}, pos2id={}, id2cap={}, stem2id={}, id2word={}, transition_counts={}, partition_sentence=False):
    ps = PorterStemmer()
    next_id = len(word2id)
    if next_id == 0:
        word2id["[PAD]"] = 0
        next_id = 1
    next_tag_id = len(tag2id)
    next_stem_id = len(stem2id)
    pos_id = len(pos2id)
    X_train = []; y_train = []; level_h_train = []; level_d_train = []
    x_chunk = [[],[],[]]; x_control = []; y_control = []; level_h = []; level_d = []

    for f in files:
        #print(f)
        with open(f) as fp:
            prev_token = None; prev_prev_token = None
            prev_tag = 7
            for line in fp:
                elems = line.split(' =======')
                #print(line)
                #print(elems)
                if elems[0] == "END-OF-CHUNK":
                    if elems[1].strip().startswith('B-REF') or elems[1].strip().startswith('I-REF'):
                        curr_tag = 'O'
                    else:
                        curr_tag = elems[1].strip()

                    if curr_tag not in tag2id:
                        tag2id[curr_tag] = next_tag_id
                        next_tag_id += 1


                    y_control.append(tag2id[curr_tag])
                    level_h.append(int(elems[2].strip()))
                    level_d.append(int(elems[3].strip()))
                    x_control.append(x_chunk)
                    x_chunk = [[],[],[]]

                    if prev_tag not in transition_counts:
                        transition_counts[prev_tag] = {}
                    if tag2id[curr_tag] not in transition_counts[prev_tag]:
                        transition_counts[prev_tag][tag2id[curr_tag]] = 0
                    transition_counts[prev_tag][tag2id[curr_tag]] += 1

                    prev_tag = tag2id[curr_tag]

                    if partition_sentence and prev_token == "." and prev_prev_token not in ["e.g", "i.e", "etc"]:
                        # Split the control statement
                        prev_prev_token = None
                        prev_token = None
                        if len(y_control) > 0:
                            y_train.append(np.array(y_control))
                            X_train.append(x_control)
                            level_h_train.append(level_h)
                            level_d_train.append(level_d)
                            curr_contr = [x[0] for x in x_control]
                            curr_contr_str = ""
                            for chunk in curr_contr:
                                curr_contr_str += " ".join([id2word[w] for w in chunk]) + " | "
                            #print(curr_contr_str)
                            #print(y_control)
                            #print("------")
                        y_control = []; x_control = []
                        level_h = []; level_d = []
                elif elems[0] == "END-OF-CONTROL":
                    prev_token = None
                    prev_prev_token = None
                    if len(y_control) > 0:
                        y_train.append(np.array(y_control))
                        X_train.append(x_control)
                        level_h_train.append(level_h)
                        level_d_train.append(level_d)
                        
                        curr_contr = [x[0] for x in x_control]
                        curr_contr_str = ""
                        for chunk in curr_contr:
                            curr_contr_str += " ".join([id2word[w] for w in chunk]) + " | "
                        
                        #print(curr_contr_str)
                        #print(y_control)
                        #print("------")
                    
                    y_control = []; x_control = []
                    level_h = []; level_d = []

                    if prev_tag not in transition_counts:
                        transition_counts[prev_tag] = {}
                    if 8 not in transition_counts[prev_tag]:
                        transition_counts[prev_tag][8] = 0
                    transition_counts[prev_tag][8] += 1
                    prev_tag = 7

                else:
                    stem = ps.stem(elems[0].lower())
                    next_id, pos_id, next_stem_id = \
                        get_token(elems, stem, word2id, next_id, pos2id, pos_id, id2cap, stem2id, next_stem_id, id2word)
                    prev_prev_token = prev_token
                    prev_token = elems[0]

                    x_chunk[0].append(word2id[elems[0]])
                    x_chunk[1].append(pos2id[elems[2]])
                    x_chunk[2].append(stem2id[stem])
    #exit()
    return X_train, np.array(y_train), level_h_train, level_d_train

def max_lengths(X, y):
    # First calculate the max
    max_chunks = 0; max_tokens = 0
    for x, _y in zip(X, y):
        max_chunks = max(max_chunks, len(_y))
        for elems in x:
            max_tokens = max(max_tokens, len(elems[0]))
    return max_chunks, max_tokens

def bert_sequences(X, y, max_chunks, max_tokens, id2word, bert_model):
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    X_new = []; y_new = []
    x_len = []; x_chunk_len = []
    #print("max_chunks", max_chunks, "max_tokens", max_tokens)

    for x, y_ in zip(X, y):
        chunks_new = []; curr_chunk_len = []
        input_ids = []; token_ids = []; attention_masks = []
        for elems in x:
            tokens = " ".join([id2word[i] for i in elems[0]])
            #print(tokens)
            encoded_input = tokenizer(tokens, padding='max_length', max_length=max_tokens, truncation=True)
            input_ids.append(encoded_input['input_ids'])
            token_ids.append(encoded_input['token_type_ids'])
            attention_masks.append(encoded_input['attention_mask'])
            #print(len(encoded_input['input_ids']), len(encoded_input['token_type_ids']), len(encoded_input['attention_mask']))
            curr_len = encoded_input['attention_mask'].count(1)
            curr_chunk_len.append(curr_len)
        x_chunk_len.append(curr_chunk_len + [0] * (max_chunks - len(curr_chunk_len)))
        #print(np.array(input_ids).shape, np.array(token_ids).shape, np.array(attention_masks).shape)
        chunks_new = np.array([input_ids, token_ids, attention_masks])
        #print(chunks_new.shape)
        n_chunks = len(input_ids)
        if n_chunks < max_chunks:
            padded_chunks = np.zeros((3, max_chunks - n_chunks, max_tokens))
            chunks_new = np.concatenate((chunks_new, padded_chunks), axis=1)

        x_len.append(n_chunks)
        X_new.append(chunks_new)
        y_new.append(np.concatenate((y_, np.negative(np.ones((max_chunks - n_chunks,))))))
    
    X_new = np.array(X_new)
    y_new = np.array(y_new)
    x_len = np.array(x_len)
    x_chunk_len = np.array(x_chunk_len)

    return X_new, y_new, x_len, x_chunk_len

def pad_sequences(X, y, max_chunks, max_tokens):
    X_new = []; y_new = []
    x_len = []; x_chunk_len = []
    for x, y_ in zip(X, y):
        chunks_new = []; curr_chunk_len = []
        for elems in x:
            tokens_new = elems[0] + [0] * (max_tokens - len(elems[0]))
            chunks_new.append(tokens_new)
            curr_chunk_len.append(len(elems[0]))
        x_chunk_len.append(curr_chunk_len + [0] * (max_chunks - len(curr_chunk_len)))
        #print(x_chunk_len[-1])
        chunks_new = np.array(chunks_new)
        (n_chunks, _) = chunks_new.shape
        x_len.append(n_chunks)
        #print(chunks_new.shape)
        if n_chunks < max_chunks:
            padded_chunks = np.zeros((max_chunks - n_chunks, max_tokens))
            #print(padded_chunks.shape)
            chunks_new = np.concatenate((chunks_new, padded_chunks))
        X_new.append(chunks_new)
        #print(max_chunks, n_chunks)
        y_new.append(np.concatenate((y_, np.negative(np.ones((max_chunks - n_chunks,))))))
    X_new = np.array(X_new)
    y_new = np.array(y_new)
    x_len = np.array(x_len)
    #print(x_len.shape)
    x_chunk_len = np.array(x_chunk_len)

    return X_new, y_new, x_len, x_chunk_len

def pad_features(X, max_chunks):
    X_new = []; y_new = []
    for x in X:
        chunks_new = []
        for chunk in x:
            chunks_new.append(chunk)
        chunks_new = np.array(chunks_new)
        (n_chunks, feat_sz) = chunks_new.shape
        if n_chunks < max_chunks:
            padded_chunks = np.zeros((max_chunks - n_chunks, feat_sz))
            chunks_new = np.concatenate((chunks_new, padded_chunks))
        X_new.append(chunks_new)
    X_new = np.array(X_new)
    return X_new

def load_glove_embeddings(word_embed_path, read_size, embedding_dim, word2id):
    embeddings = np.zeros((len(word2id), embedding_dim))
    unk_words = 0
    unknowns = []
    dict_word2vec = {}
    with open(word_embed_path) as f:
        count = 0
        while count < read_size:
            line = f.readline().split()
            word = line[0]
            vec = line[1:]
            dict_word2vec[word] = vec
            count = count + 1
        for (token, index) in word2id.items():
            if token in dict_word2vec:
                embeddings[index] = dict_word2vec[token]
            elif token != "[PAD]":
                unk_words += 1
                embeddings[index] = np.random.rand(embedding_dim)
                unknowns.append(token)
    print("GLOVE: vocabulary: " + str(len(embeddings)), "unk_words: " + str(unk_words))
    tensor = torch.from_numpy(embeddings).float()
    return tensor

def load_network_embeddings(word_embed_path, embedding_dim, word2id):
    embeddings = np.zeros((len(word2id), embedding_dim))
    unk_words = 0
    unknowns = []
    dict_word2vec = {}
    with open(word_embed_path) as f:
        line = f.readline()
        while line:
            line = line.split()
            word = line[0]
            vec = line[1:]
            dict_word2vec[word] = vec
            line = f.readline()
    for (token, index) in word2id.items():
        if token in dict_word2vec.keys():
            embeddings[index] = dict_word2vec[token]
        elif token != "[PAD]":
            unk_words += 1
            embeddings[index] = np.random.rand(embedding_dim)
            unknowns.append(token)
            # print("GLOVE UNKNOWN: " + str(token))
    print("NETWORK: vocabulary: " + str(len(embeddings)), "unk_words: " + str(unk_words))
    tensor = torch.from_numpy(embeddings).float()
    return tensor

def any_transition_dirs(tokens):
    transition_dirs = r'to|from|through|for'
    for token in tokens:
        if re.match(transition_dirs, token):
            return True
    return False

def any_transition_verbs(tokens):
    transition_verbs = r'enter.*|mov.*|chang.*|stay.*|leav.*|go.*|remain*|return*'
    tokens_str = " ".join(tokens)

    for token in tokens:
        if re.match(transition_verbs, token):
            return True
    if re.match(r'(.*\w\.state :=.*)|(.*\w\.state = .*)', tokens_str):
        return True

    return False

def any_action_verbs(tokens):
    action_verbs = r'send.*|receiv.*|issu.*|generat.*|form.*|creat.*'
    tokens_str = " ".join(tokens)
    for token in tokens:
        if re.match(action_verbs, token):
            return True

    return False

def any_defs(tokens, def_states):
    for token in tokens:
        #if overlap(token, def_states):
        if token in def_states and def_states[token] != "None":
            return True
    return False

def any_verbs(tokens, pos_tags, defs):
    for token, tag in zip(tokens, pos_tags):
        if token not in defs and tag[1].startswith('VB'):
            #print("FOUND!--", token, tag[1])
            return True
    return False

def heuristic_transitions(X_test_data, y_test_trans, y_pred_trans, id2word, def_states):
    '''
        Apply some heuristics:
        1) If there is a transition verb and a state name, flip it to be a transition
        2) If there is a transition directory word (to, from, through), and a state name, flip it to be a transition
    '''
    new_pred_trans = []
    for x, y_g, y_p in zip(X_test_data, y_test_trans, y_pred_trans):
        new_y_p = []

        for span_x, span_y in zip(x, y_p):
            span_words = [id2word[i].lower() for i in span_x]
            # Check for heuristic 1
            if any_transition_verbs(span_words) and any_defs(span_words, def_states) and (not span_y[0].endswith('TRANSITION')):
                if "if" in span_words and "enter" in span_words:
                    # this was there on previous iteration, TO-DO: make it cleaner
                    enter_index = span_words.index("enter")
                    new_span = span_y[:enter_index] + ['B-TRANSITION'] + ['I-TRANSITION'] * (len(span_y) - 1 - enter_index)
                    #print(span_words)
                    #print(new_span)
                else:
                    new_span = ['B-TRANSITION'] + ['I-TRANSITION'] * (len(span_y) - 1)
                #print("Heuristic 1 -- ", span_words, new_span)
                new_y_p.append(new_span)
            # Check for heuristic 2
            elif any_transition_dirs(span_words) and any_defs(span_words, def_states) and (not span_y[0].endswith('TRANSITION')) and "if" not in span_words:
                new_span = ['B-TRANSITION'] + ['I-TRANSITION'] * (len(span_y) - 1)
                #print("Heuristic 2 -- ", span_words, new_span)
                new_y_p.append(new_span)
            else:
                new_y_p.append(span_y)

        new_pred_trans.append(new_y_p)

    return new_pred_trans

def get_first_word_index(tokens):
    for i in range(0, len(tokens)):
        if tokens[i] != "&newline;":
            return i
    return -1

def heuristic_actions(X_test_data, y_test_trans, y_pred_trans, id2word, def_states, def_events):
    '''
        Apply some heuristics:
        1) If there is an action verb and an event name (and no transition signal), flip it to be an action
    '''
    new_pred_trans = []
    for x, y_g, y_p in zip(X_test_data, y_test_trans, y_pred_trans):
        new_y_p = []

        for span_x, span_y in zip(x, y_p):
            span_words = [id2word[i].lower() for i in span_x]
            
            # Check for heurstic 1
            if any_action_verbs(span_words) and any_defs(span_words, def_events) and (not span_y[0].endswith('ACTION')) and "if" not in span_words:
                #and not (any_defs(span_words, def_states) and (any_transition_verbs(span_words) or any_transition_dirs(span_words))):
                #print(span_words))
                #print("HERE", span_words)
                new_span = ['B-ACTION'] + ['I-ACTION'] * (len(span_y) - 1)
                new_y_p.append(new_span)

                if any_defs(span_words, def_states) or (any_transition_verbs(span_words) or any_transition_dirs(span_words)):
                    print(span_words)
            else:
                new_y_p.append(span_y)

        new_pred_trans.append(new_y_p)

    return new_pred_trans


def heuristic_errors(X_test_data, y_test_trans, y_pred_trans, id2word):
    pass

def heuristic_outside(X_test_data, y_test_trans, y_pred_trans, id2word, def_states, def_events, def_variables):
    '''
        Apply a heuristic:
        1) If we have OUTSIDE spans and they have mentions to events and a verb, flip it to be an action
        2) If we have OUTSIDE spans and they have mentions to states or events, flip it to be a trigger
        3) If we have a span with references to states, and no transitions verbs, flip it to be a trigger
        
        (SCRATCHING THIS ONE) ---If we have a trigger with no references, flip it to be outside to avoid unnecesary recursions
    '''
    new_pred_trans = []
    all_defs = set(def_states.keys()) | set(def_events.keys())

    for x, y_g, y_p in zip(X_test_data, y_test_trans, y_pred_trans):
        new_y_p = []

        for span_x, span_y in zip(x, y_p):
            span_words = [id2word[i].lower() for i in span_x]
            #doc = nlp(" ".join(span_x))
            #pos_tags = [token.pos_ for token in doc]
            pos_tags = nltk.pos_tag(span_words)
            if any_defs(span_words, def_events) and span_y[0] == "O" and any_verbs(span_words, pos_tags, all_defs):
                #print(span_words)
                new_span = ['B-ACTION'] + ['I-ACTION'] * (len(span_y) - 1)
                new_y_p.append(new_span)
            elif any_defs(span_words, def_states) and span_y[0] == "O":
                new_span = ['B-TRIGGER'] + ['I-TRIGGER'] * (len(span_y) - 1)
                new_y_p.append(new_span)
            elif any_defs(span_words, def_states) and not any_transition_verbs(span_words) and not any_transition_dirs(span_words)\
                 and not (span_words[0] == "note" and span_words[1] == "that") \
                 and not (span_words[0] == "in" and span_words[1] == "addition" and span_words[2] == "to"):
                #and not span_y[0].endswith('TRIGGER') \
                #print("HERE", span_words)
                new_span = ['B-TRIGGER'] + ['I-TRIGGER'] * (len(span_y) - 1)
                new_y_p.append(new_span)
            elif any_defs(span_words, def_events) and ("call" in span_words or "timeout" in span_words) and\
                not (any_defs(span_words, def_states) and (any_transition_dirs(span_words) or any_transition_verbs(span_words))):
                new_span = ['B-TRIGGER'] + ['I-TRIGGER'] * (len(span_y) - 1)
                new_y_p.append(new_span)
            elif (
                    (span_words[0] == "in" and span_words[1] == "addition" and span_words[2] == "to") \
                  ) and "otherwise" not in span_words and "if" not in span_words:
                new_span = ['O'] * len(span_y)
                new_y_p.append(new_span)
            elif span_y[0] == "O" and 'timer' in span_words:
                new_span = ['B-TIMER'] + ['I-TIMER'] * (len(span_y) - 1)
                new_y_p.append(new_span)
            elif span_y[0] == "O" and 'error' in span_words:
                new_span = ['B-ERROR'] + ['I-ERROR'] * (len(span_y) - 1)
                new_y_p.append(new_span)
            elif span_y[0] == "O" and any_defs(span_words, def_variables):
                new_span = ['B-VARIABLE'] + ['I-VARIABLE'] * (len(span_y) - 1)
                new_y_p.append(new_span)
            else:
                new_y_p.append(span_y)
            '''
            elif (((not any_defs(span_words, def_events)) and (not any_defs(span_words, def_states))) or \
                    (span_words[0] == "note" and span_words[1] == "that") or \
                    (span_words[0] == "in" and span_words[1] == "addition" and span_words[2] == "to") \
                  ) and "otherwise" not in span_words and "if" not in span_words:
                new_span = ['O'] * len(span_y)
                new_y_p.append(new_span)
            '''

        new_pred_trans.append(new_y_p)

    return new_pred_trans

def join_consecutive_transitions(X_test_data, y_test_trans, y_pred_trans, level_h, level_d):
    '''
       Turn consecutive transition spans into a single span 
    '''
    new_pred_trans = []
    for x, y_g, y_p, h, d in zip(X_test_data, y_test_trans, y_pred_trans, level_h, level_d):
        new_y_p = []

        new_y_p.append(y_p[0])

        for i in range(1, len(y_p)):

            if y_p[i-1][0].endswith('TRANSITION') and y_p[i][0].endswith('TRANSITION') and \
               h[i-1][0] == h[i][0] and d[i-1][0] == d[i][0]:
                new_span = ['I-TRANSITION'] * len(y_p[i])
                new_y_p.append(new_span)
            else:
                new_y_p.append(y_p[i])

        new_pred_trans.append(new_y_p)

    return new_pred_trans

def same_span(tag, prev_tag):
    if tag and prev_tag and tag[2:] == prev_tag[2:] and tag[0] == 'I':
        return True
    return False

def write_definitions(def_states, def_events):
    ret_str = ""
    for state in def_states:
        ret_str += '<def_state id="{}">{}</def_state>\n'.format(def_states[state], state)
    for event in def_events:
        if event in ['acknowledgment', 'reset']:
            continue
        ret_str += '<def_event id="{}">{}</def_event>\n'.format(def_events[event], event)
    return ret_str

def tag_lookahead(index, x_trans, y_p, current_tag):
    j = index + 1; tag_str = x_trans[index] + " "
    if j < len(y_p):
        current_tag_lookahead = y_p[j]
        while (same_span(current_tag_lookahead, current_tag)):
            tag_str += x_trans[j] + " "
            j += 1
            if j >= len(y_p):
                break
            current_tag_lookahead = y_p[j]
    tag_str = tag_str.replace("&newline;", "")
    return tag_str

def find_acknowledgment(tag_str, type_ref):
    tag_split = tag_str.split()
    if 'acked' in tag_split:
        ack_word_index = tag_split.index('acked')
    elif 'acknowledged' in tag_split:
        ack_word_index = tag_split.index('acknowledged')
    else:
        # In case of acknowledgement of ... take the whole phrase
        ack_word_index = len(tag_split) - 1
    tag_split = tag_split[:ack_word_index+1]
    ack_tags = ['B-ACK'] + ['I-ACK'] * (len(tag_split) - 1)
    ack_type = type_ref
    #print(tag_split)
    return ack_tags, ack_type

def last_was_trigger(seq_tags, seq_words):
    rev_tags = seq_tags[::-1];
    rev_words = seq_words[::-1];
    i = 0
    while (i < len(rev_tags) and rev_tags[i] == 'O' and (rev_words[i] == "&newline;" or rev_words[i].isspace())):
        i += 1
    if i < len(rev_tags) and rev_tags[i].endswith('TRIGGER'):
        return True
    return False

def guess_recursive_controls(index, y_p, x_trans, current_tag, offset, offset_str, open_recursive_control, open_continued_control):
    ret_str = ""
    if current_tag[2:].lower() == "trigger": #and x_trans[index:][0] != "otherwise":
        num_triggers = len([p for p in y_p[index:] if p.endswith('TRIGGER')])
        if num_triggers != len(y_p[index:]):
            #print(num_trigers)
            #print([p for p in y_p[index:] if p.endswith('TRIGGER')])
            #print(y_p[index:])
            #print(x_trans[index:])
            #print(x_trans[index-1])
            #print(num_triggers, len(y_p[index:]))
            #if open_continued_control:
            #    ret_str += offset_str + "".join(["\t"] * offset) + "\n</control_GR>"
            #    open_continued_control = False
            if not ret_str.endswith('<control relevant="true">'):
                if not last_was_trigger(y_p[:index], x_trans[:index]):
                    offset_str = "".join(["\t"] * offset)
                    if open_continued_control:
                        ret_str += '\n' + offset_str + "</control>"
                    ret_str += '\n' + offset_str + '<control relevant="true">'
                    open_continued_control = True
                else:
                    offset += 1
                    offset_str = "".join(["\t"] * offset)
                    ret_str += '\n' + offset_str + '<control relevant="true">'
                    open_recursive_control = True

    return open_recursive_control, open_continued_control, offset_str, ret_str, offset

def srl_events(predictor, action_str, current_tag, def_states, protocol):
    tag_type = None; tags = []
    if not re.match('.*(syn-sent|syn-received).*', action_str):
        srl = predictor.predict(sentence=action_str)
        #print(srl['words'])
        #print('------')

        # don't break and keep the last observed verb (are they in order? make sure!)
        for verb in srl['verbs']:
            if verb['verb'].startswith('send') or verb['verb'].startswith('sent'):
                tags = verb['tags']
                tag_type = "send"
                break
            elif verb['verb'].startswith('receiv'):
                tags = verb['tags']
                tag_type = "receive"
                break
            elif verb['verb'].startswith('issu') or verb['verb'].startswith('form') or verb['verb'].startswith('generat'):
                ## WARNING -- cheating!
                #if "reset" in action_str:
                #    pass
                #else:
                tags = verb['tags']
                if "send" in action_str or "sent" in action_str and "receiv" not in action_str:
                    tag_type = "send"
                    print("send", action_str)
                elif "send" in action_str:
                    print("HERE", action_str)
                elif "receive" in action_str and "send" not in action_str and "receiv" not in action_str:
                    tag_type = "receive"
                    print("receive", action_str)
                else:
                    tag_type = "issue"
                #print(action_str)
                break
            #elif verb['verb'].startswith('ack'):
            #    tag_type = "receive"
            elif 'handshake' in srl['words']:
                tag_type = "send"
            #elif current_tag.endswith("ACTION") and any_defs(action_str.split(), def_states):
            #else:
            #    print("RECEIVE", action_str)
            #    print(action_str, current_tag)
            #    tag_type = "receive"
            #    #print(verb['verb'])
            #    pass
            '''
            elif "send" in action_str or "sent" in action_str and "receiv" not in action_str:
                print("SEND", action_str)
                tag_type = "send"
            elif "receiv" in action_str and "send" not in action_str and "sent" not in action_str:
                print("RECEIVE", action_str)
                tag_type = "receive"
            '''
    if tag_type is None and current_tag.endswith('ACTION') and "do not" not in action_str and "do n ` t" not in action_str:
        #print("\tRECEIVE", action_str)
        tag_type = "receive"

    return tags, tag_type

def srl_transitions(predictor, transition_str, def_states):
    tag_type = None; tags = []
    srl = predictor.predict(sentence=transition_str)
    #print(transition_str)
    #print(srl)
    #print("-------")
    for verb in srl['verbs']:
        if verb['verb'].startswith('mov') or \
           verb['verb'].startswith('enter') or \
           verb['verb'].startswith('transition') or \
           verb['verb'].startswith('go') or \
           verb['verb'].startswith('chang') or \
           verb['verb'].startswith('leav') or \
           verb['verb'].startswith('remain') or\
           verb['verb'].startswith('stay'):

            tags = verb['tags']

            if verb['verb'].startswith('leav'):
                print(srl)

            break

    transition_splits = transition_str.split()
    '''
    if transition_splits[0] in ["to", "from"] and transition_splits[1] in def_states and len(tags) == 0:
        tags = ['O'] * len(transition_splits)
        tags[0] = 'B-ARGM-DIR'; tags[1] = "I-ARGM-DIR"
    '''
    return tags, tag_type

def overlap(word, definitions):
    splits = word.split('-')
    if len(splits) == 2:
        word_sp = splits[1]
    else:
        word_sp = splits[0]

    for _def in definitions:
        def_splits = _def.split('-')
        if len(def_splits) == 2:
            def_sp = def_splits[1]
        else:
            def_sp = def_splits[0]

        if word == _def or word == def_sp or word_sp == _def or word_sp == def_sp:
            definitions[word] = definitions[_def]
            print(word, _def)
            return True
    return False


def write_results(X_test_data, y_test_trans, y_pred_trans, level_h_trans, level_d_trans, id2word, def_states, def_events, def_events_constrained, protocol): # , cuda_device):
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz") #, cuda_device=cuda_device)
    
    ret_str = "<p>"
    
    # Write down definitions
    #print(def_events_constrained)
    ret_str += write_definitions(def_states, def_events_constrained)

    # Iterate over sequences
    n_controls = len(y_pred_trans)
    pbar = tqdm(total=n_controls)

    for x, y_g, y_p, h, d in zip(X_test_data, y_test_trans, y_pred_trans, level_h_trans, level_d_trans):

        cur_str =  '\n<control relevant="true">'
        ret_str += '\n<control relevant="true">'
        
        x_trans = [id2word[i].lower() for i in x]

        #print("H", [(a,b,c) for (a,b,c) in zip(x_trans, h, y_p)])
        #print("////////////")
        #print("D", [(a,b,c) for (a,b,c) in zip(x_trans, d, y_p)])
        #exit()

        #open_recursive_control = False
        #open_continued_control = False
        prev_tag = None
        prev_h = 0; prev_d = 0; prev_offset = 1
        num_open_control = {1: 1}
        i = 0
        ack_tags = []; ack_type = None
        open_arg = False;
        explicit_source = False; explicit_target = False; explicit_intermediate = False
        explicit_state = False; explicit_type = False

        tag_type = None; tags = []

        while (i < len(x_trans)):
            offset = d[i];
            #print(x_trans[i], offset)

            word = x_trans[i]
            pred_tag = y_p[i]

            current_tag = pred_tag
            if prev_tag is not None and not same_span(current_tag, prev_tag) and prev_tag != 'O':
                if prev_tag[2:].lower() == 'action' and open_arg:
                    ret_str += "</arg> "; open_arg = False
                    cur_str += "</arg> "; open_arg = False
                    tag_type = None
                
                offset_str = "".join(['\t'] * offset)
                ret_str += offset_str + "</{}>".format(prev_tag[2:].lower())
                cur_str += offset_str + "</{}>".format(prev_tag[2:].lower())
                closed = True

            if current_tag is None or (not same_span(current_tag, prev_tag) and current_tag != 'O'):

                '''
                # Guess recursive controls
                open_recursive_control, open_continued_control, offset_str, new_ret_str, offset =\
                    guess_recursive_controls(i, y_p, x_trans, current_tag, offset, offset_str,
                                             open_recursive_control, open_continued_control)
                ret_str += new_ret_str
                '''
                # Use offsets to identify recursive controls
                if prev_h > 0 and prev_d > 0 and \
                   (h[i] > prev_h and prev_d == d[i]):
                    offset_str = "".join(['\t'] * (offset-1))
                    ret_str += "\n" + offset_str + '</control>'
                    cur_str += "\n" + offset_str + '</control>'
                    #print("------")
                    #print(cur_str)
                    #print("removing", offset, num_open_control)
                    num_open_control[offset] -= 1

                if prev_h > 0 and prev_d > 0 and \
                   (d[i] < prev_d):
                    if prev_h > 0 and h[i] > prev_h:
                        relevant_offsets = [index for index in num_open_control if num_open_control[index] > 0 and index >= offset]
                    else:
                        relevant_offsets = [index for index in num_open_control if num_open_control[index] > 0 and index > offset]
                    relevant_offsets.sort(reverse=True)
                    for open_control in relevant_offsets:
                        offset_str = "".join(['\t'] * (open_control-1))
                        ret_str += "\n" + offset_str + "</control>"
                        cur_str += "\n" + offset_str + "</control>"
                        num_open_control[open_control] -= 1

                if prev_h > 0 and h[i] > prev_h:
                    offset_str = "".join(['\t'] * (offset-1))
                    ret_str += "\n\n" + offset_str + '<control relevant="true">'
                    cur_str += "\n\n" + offset_str + '<control relevant="true">'
                    if offset not in num_open_control:
                        num_open_control[offset] = 0
                    num_open_control[offset] += 1
                    #print("adding", offset, num_open_control)
                tag_type = None; tags = []

                #if current_tag[2:].lower() in ["action", "trigger"]:
                # Do a look-ahead
                action_str = tag_lookahead(i, x_trans, y_p, current_tag)
                tags, tag_type = srl_events(predictor, action_str, current_tag, def_states, protocol)
                
                if current_tag[2:].lower() == "transition":
                    # Do a look-ahead
                    transition_str = tag_lookahead(i, x_trans, y_p, current_tag)
                    tags, tag_type = srl_transitions(predictor, transition_str, def_states)

                if tag_type is None or current_tag[2:].lower() != "action":
                    offset_str = "".join(['\t'] * offset)
                    ret_str += "\n" + offset_str + "<{}>".format(current_tag[2:].lower())
                    cur_str += "\n" + offset_str + "<{}>".format(current_tag[2:].lower())
                else:
                    offset_str = "".join(['\t'] * offset)
                    ret_str += "\n" + offset_str + '<{} type="{}">'.format(current_tag[2:].lower(), tag_type)
                    cur_str += "\n" + offset_str + '<{} type="{}">'.format(current_tag[2:].lower(), tag_type)
                closed = False

            word = word.replace("&newline;", "")
            word = word.replace("&", "&amp;")
            word = word.replace("<", "&lt;")
            word = word.replace(">", "&gt;")

            if word.endswith("type"):
                explicit_type = True
            if word.endswith("state"):
                explicit_state = True

            # Add opening tag for identified argument in case of actions
            if current_tag[2:].lower() == 'action' and len(tags) > 0 and tags[0] == 'B-ARG1':
                ret_str += "<arg> "
                cur_str += "<arg> "
                open_arg = True

            # Check explicit source/target in transition tags
            if current_tag[2:].lower() == 'transition' and len(tags) > 0 and (tags[0] in ['B-ARGM-DIR', 'B-ARG2', 'B-ARG1', 'I-ARGM-DIR', 'I-ARG2', 'I-ARG1', 'B-ARGM-PRD', 'I-ARGM-PRD']) and word in ["to", "for"]:
                explicit_target = True
                #print("TO | ", transition_str, tags)

            if current_tag[2:].lower() == 'transition' and len(tags) > 0 and (tags[0] in ['B-ARGM-DIR', 'B-ARG2', 'B-ARG1', 'I-ARGM-DIR', 'I-ARG2', 'I-ARG1', 'B-ARGM-PRD', 'I-ARGM-PRD']) and word == "from":
                explicit_source = True

            if current_tag[2:].lower() == 'transition' and len(tags) > 0 and tags[0] in ['B-V', 'I-V'] and word.startswith('leav'):
                explicit_source = True

            if current_tag[2:].lower() == 'transition' and len(tags) > 0 and (tags[0] in ['B-ARGM-DIR', 'B-ARG2', 'B-ARG1', 'I-ARGM-DIR', 'I-ARG2', 'I-ARG1', 'B-ARGM-PRD', 'I-ARGM-PRD']) and word == "through":
                explicit_intermediate = True
                #print("FROM | ", transition_str, tags)
            
            #if current_tag[2:].lower() in ['trigger', 'transition'] and overlap(word, def_states):
            if current_tag[2:].lower() in ['trigger', 'transition'] and word in def_states and (not explicit_type):
                tagged_word = ""

                # If source/target explicit, open tag
                if explicit_target:
                    tagged_word += "<arg_target>"
                if explicit_source:
                    tagged_word += "<arg_source>"
                if explicit_intermediate:
                    tagged_word += "<arg_intermediate>"

                # Write the state
                tagged_word += '<ref_state id="{}">{}</ref_state>'.format(def_states[word], word)
                explicit_type = False

                # If source/target explicit, close the tag and turn off the flags
                if explicit_source:
                    tagged_word += "</arg_source>"
                if explicit_target:
                    tagged_word += "</arg_target>"
                if explicit_intermediate:
                   tagged_word += "</arg_intermediate>" 
                explicit_source = False; explicit_target = False; explicit_intermediate = False;
                word = tagged_word
                #print(word)

            if word.endswith('msl') and 'timeout' in def_events:
                word = '<ref_event id="{}" type="compute">{}</ref_event>'.format(def_events['timeout'], word)
            
            #elif current_tag[2:].lower() in ['trigger', 'action'] and overlap(word, def_events) and word not in ["send", "receive"]:
            elif word in def_events and word not in ["send", "receive"]:
                
                # Do a lookahead to see if we have an acknowledgment phrase
                if len(ack_tags) > 0 and ack_tags[0] in ['B-ACK', 'I-ACK']:
                    pass
                else:
                    tag_str = tag_lookahead(i, x_trans, y_p, current_tag)
                    type_ref = tag_type
                    
                    if type_ref == "issue":
                        type_ref = "send"
                    if current_tag[2:].lower() == 'trigger' and type_ref is None:
                        type_ref = "receive"
                    if re.match('[\s\w]*( acked| acknowledged|acknowledgment of|acknowledge).*', tag_str) and protocol == "TCP":
                        ack_tags, ack_type = find_acknowledgment(tag_str, type_ref)
                        print(tag_str, ack_tags, tags, ack_type)
                    else:
                        word = '<ref_event id="{}" type="{}">{}</ref_event>'.format(def_events[word], type_ref, word)
                        explicit_type = False

            if current_tag[2:].lower() in ['action', 'trigger'] and len(ack_tags) > 0 and ack_tags[0] == 'B-ACK':
                ret_str += '<ref_event id="{}" type="{}">'.format(def_events['ack'], ack_type, word)
                cur_str += '<ref_event id="{}" type="{}">'.format(def_events['ack'], ack_type, word)
            
            # Add the actual word
            ret_str += word + " "
            cur_str += word + " "

            if current_tag[2:].lower() in ['action', 'trigger'] and len(ack_tags) == 1 and ack_tags[0] == 'I-ACK':
                ret_str += '</ref_event>'
                cur_str += '</ref_event>'

            '''
                Deal with action cases below
                We need to identify all the cases in which we are closing the ARG 
            '''
            if current_tag[2:].lower() == 'action' and len(tags) == 1 and tags[0] == 'I-ARG1':
                ret_str += "</arg> "; open_arg = False
                cur_str += "</arg> "; open_arg = False
                #tag_type = None
                #print("Closing 1", tags[0], tags)
            elif current_tag[2:].lower() == 'action' and len(tags) == 2 and word == '``' and tags[0] == 'I-ARG1':
                ret_str += "</arg> "; open_arg = False
                cur_str += "</arg> "; open_arg = False
                #tag_type = None
            elif current_tag[2:].lower() == 'action' and len(tags) == 3 and re.match(r'\w+(\=|\-|\/)\w+', word) and tags[0] == 'I-ARG1':
                ret_str += "</arg> "; open_arg = False
                cur_str += "</arg> "; open_arg = False
                #tag_type = None
            elif current_tag[2:].lower() == 'action' and len(tags) > 1 and tags[0] == 'I-ARG1' and tags[1] != 'I-ARG1':
                ret_str += "</arg> "; open_arg = False
                cur_str += "</arg> "; open_arg = False
                #tag_type = None
                #print("Closing 2", tags[0], tags)
            elif current_tag[2:].lower() == 'action' and len(tags) == 1 and tags[0] == 'B-ARG1':
                ret_str += "</arg> "; open_arg = False
                cur_str += "</arg> "; open_arg = False
                #tag_type = None
                #print("Closing 3", tags[0], tags)
            elif current_tag[2:].lower() == 'action' and len(tags) == 2 and word == '``' and tags[0] == 'B-ARG1':
                ret_str += "</arg> "; open_arg = False
                cur_str += "</arg> "; open_arg = False
                #tag_type = None
            elif current_tag[2:].lower() == 'action' and len(tags) == 3 and re.match(r'\w+(\=|\-|\/)\w+', word) and tags[0] == 'B-ARG1':
                ret_str += "</arg> "; open_arg = False
                cur_str += "</arg> "; open_arg = False
                #tag_type = None
            elif current_tag[2:].lower() == 'action' and len(tags) > 1 and tags[0] == 'B-ARG1' and tags[1] != "I-ARG1":
                ret_str += "</arg> "; open_arg = False
                cur_str += "</arg> "; open_arg = False
                #tag_type = None
                #print("Closing 4", tags[0], tags)


            # Move two positions to account for the fact that the SRL separates `` into two tokens
            # It also happens when we have a composed word with symbols in the middle
            if len(tags) > 0 and word == '``':
                tags = tags[2:]
            elif len(tags) > 0 and re.match(r'\w+(\=|\-|\/)\w+', word):
                tags = tags[3:]
            elif len(tags) > 0:
                tags = tags[1:]

            if len(ack_tags) > 0:
                ack_tags = ack_tags[1:]

            prev_tag = current_tag
            prev_h = h[i]
            if current_tag != 'O':
                prev_d = d[i]
                prev_offset = offset
            i += 1

        if not closed and prev_tag != 'O':
            if prev_tag[2:].lower() == 'action' and open_arg:
                ret_str += "</arg> "; open_arg = False
                cur_str += "</arg> "; open_arg = False
                #tag_type = None
            offset_str = "".join(['\t'] * offset)
            ret_str += offset_str + "</{}>".format(prev_tag[2:].lower())
            cur_str += offset_str + "</{}>".format(prev_tag[2:].lower())

        relevant_offsets = [index for index in num_open_control if num_open_control[index] > 0]
        relevant_offsets.sort(reverse=True)

        for open_control in relevant_offsets:
            offset_str = "".join(['\t'] * (open_control-1))
            ret_str += "\n" + offset_str + "</control>"
            cur_str += "\n" + offset_str + "</control>"
        num_open_control[open_control] -= 1

        ret_str += "\n"
        cur_str += "\n"
        pbar.update(1)
        #print(cur_str)
        #print("---------------")

        #exit()
        #print(num_recursive_controls, '----------')
    pbar.close()

        #if ret_str.endswith("<error>. </control>\n"):
        #    print(closed, prev_tag)
        #    exit()
    #print(ret_str)
    #exit()
    ret_str += "</p>"
    #ret_str = ret_str.replace("\f", "").replace("\t", "")
    return ret_str

