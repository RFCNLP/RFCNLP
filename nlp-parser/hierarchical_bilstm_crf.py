import os
import torch.nn as nn
import torch
import numpy as np
import random
import argparse
from tqdm import tqdm, trange
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import *
from sklearn.preprocessing import normalize
from ner_evaluation.ner_eval import Evaluator

import data_utils
import features

START_TAG = 7
STOP_TAG = 8

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class HierarchicalBiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, token_embedding_dim, pretrained,
                 token_hidden_dim, chunk_hidden_dim, max_chunk_len, max_seq_len,
                 feat_sz, batch_size, output_dim, use_features=False):
        super(HierarchicalBiLSTM_CRF, self).__init__()
        self.token_hidden_dim = token_hidden_dim
        self.chunk_hidden_dim = chunk_hidden_dim
        self.max_chunk_len = max_chunk_len
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.output_dim = output_dim

        self.embedding = nn.Embedding(vocab_size, token_embedding_dim)
        self.embedding.weight = nn.Parameter(pretrained)

        # this has to grow if features are being considered
        dim = token_embedding_dim

        self.token_lstm = nn.LSTM(dim, token_hidden_dim, batch_first=True,
                                  bidirectional=True, num_layers=1)

        self.use_features = use_features
        if not use_features:
            self.chunk_lstm = nn.LSTM(2*token_hidden_dim, chunk_hidden_dim, batch_first=True, bidirectional=True, num_layers=1)
        else:
            self.chunk_lstm = nn.LSTM(2*token_hidden_dim + feat_sz, chunk_hidden_dim, batch_first=True, bidirectional=True, num_layers=1)

        self.fc = nn.Linear(2 * chunk_hidden_dim, output_dim + 2)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
                torch.randn(output_dim + 2, output_dim + 2))

        # These two statements enforce the constraint that we never transfer
         # to the start tag and we never transfer from the stop tag
        self.transitions.data[START_TAG, :] = -10000
        self.transitions.data[:, STOP_TAG] = -10000

        self.hidden_token_bilstm = self.init_hidden_token_bilstm()
        self.hidden_chunk_bilstm = self.init_hidden_chunk_bilstm()

    def load_transition_priors(self, transition_priors):
        with torch.no_grad():
            self.transitions.copy_(torch.from_numpy(transition_priors))

    def init_hidden_token_bilstm(self):
        var1 = torch.autograd.Variable(torch.zeros(2, self.batch_size * self.max_chunk_len, self.token_hidden_dim)).cuda()
        var2 = torch.autograd.Variable(torch.zeros(2, self.batch_size * self.max_chunk_len, self.token_hidden_dim)).cuda()
        return var1, var2

    def init_hidden_chunk_bilstm(self):
        var1 = torch.autograd.Variable(torch.zeros(2, self.batch_size, self.chunk_hidden_dim)).cuda()
        var2 = torch.autograd.Variable(torch.zeros(2, self.batch_size, self.chunk_hidden_dim)).cuda()
        return var1, var2

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.output_dim + 2), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][START_TAG] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas.cuda()

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.output_dim + 2):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                 emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.output_dim + 2)
                 # the ith entry of trans_score is the score of transitioning to
                 # next_tag from i
                 trans_score = self.transitions[next_tag].view(1, -1)
                 # The ith entry of next_tag_var is the value for the
                 # edge (i -> next_tag) before we do log-sum-exp
                 next_tag_var = forward_var + trans_score +  emit_score
                 # The forward variable for this tag is log-sum-exp of all the
                 # scores.
                 alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[STOP_TAG]
        alpha = log_sum_exp(terminal_var)
        return alpha


    def _get_bilstm_features(self, x, x_feats, x_len, x_chunk_len):
        self.batch_size = x.shape[0]

        self.hidden_token_bilstm = self.init_hidden_token_bilstm()
        self.hidden_chunk_bilstm = self.init_hidden_chunk_bilstm()

        x = self.embedding(x)
        # unsqueeze all the chunks
        x = x.view(self.batch_size * self.max_chunk_len, self.max_seq_len, -1)
        # clamp everything to minimum length of 1, but keep the original variable to mask the output later
        x_chunk_len = x_chunk_len.view(-1)
        x_chunk_len_clamped = x_chunk_len.clamp(min=1, max=self.max_seq_len).cpu()

        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_chunk_len_clamped, batch_first=True, enforce_sorted=False)
        x, self.hidden_token_bilstm = self.token_lstm(x, self.hidden_token_bilstm)
        # unpack
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # extract last timestep, since doing [-1] would get the padded zeros
        idx = (x_chunk_len_clamped - 1).view(-1, 1).expand(x.size(0), x.size(2)).unsqueeze(1).cuda()
        x = x.gather(1, idx).squeeze()

        # revert back to (batch_size, max_chunk_len)
        x = x.view(self.batch_size, self.max_chunk_len, -1)
        if self.use_features:
            #print(x.shape, x_feats.shape)
            x = torch.cat((x, x_feats), 2)

        x_len = x_len.cpu()
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x, self.hidden_chunk_bilstm = self.chunk_lstm(x, self.hidden_chunk_bilstm)
        # unpack
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=self.max_chunk_len)
        x = self.fc(x)
        return x

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).cuda()
        temp = torch.tensor([START_TAG], dtype=torch.long).cuda()

        tags = torch.cat([temp, tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[STOP_TAG, tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.output_dim + 2), -10000.)
        init_vvars[0][START_TAG] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars.cuda()
        #print("feats.size()", feats.size())

        for feat_idx, feat in enumerate(feats):
            #print(feat_idx)
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.output_dim + 2):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[STOP_TAG]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == START_TAG  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, x, x_feats, x_len, x_chunk_len, y):
        loss_accum = torch.autograd.Variable(torch.FloatTensor([0])).cuda()

        feats = self._get_bilstm_features(x, x_feats, x_len, x_chunk_len)
        n = 0

        #print(x.size(), x_len.size(), feats.size(), y.size())
        #print(x_len)

        for x_len_i, sent_feats, tags in zip(x_len, feats, y):
            #print(sent_feats[:x_len_i].size(), tags[:x_len_i].size())
            forward_score = self._forward_alg(sent_feats[:x_len_i])
            gold_score = self._score_sentence(sent_feats[:x_len_i], tags[:x_len_i])
            loss_accum += forward_score - gold_score
            n += 1
        return loss_accum / n

    def predict_sequence(self, x, x_feats, x_len, x_chunk_len):
        # Get the emission scores from the BiLSTM
        #print("x.shape", x.shape)
        lstm_feats = self._get_bilstm_features(x, x_feats, x_len, x_chunk_len)
        outputs = []
        for x_len_i, sequence in zip(x_len, lstm_feats):
            #print("sequence.shape", sequence[:x_len_i].shape)
            # Find the best path, given the features.
            score, tag_seq = self._viterbi_decode(sequence[:x_len_i])
            outputs.append(tag_seq)

        return outputs

    def forward(self, x, x_feats, x_len, x_chunk_len):  # dont confuse this with _forward_alg above
        output = self._get_bilstm_features(x, x_feats, x_len, x_chunk_len)
        return output


def evaluate(model, test_dataloader, classes):
    model.eval()
    total_loss_dev = 0
    preds = []; labels = []
    for x, x_feats, x_len, x_chunk_len, y in test_dataloader:
        x = x.cuda()
        x_feats = x_feats.cuda()
        x_len = x_len.cuda()
        x_chunk_len = x_chunk_len.cuda()
        y = y.cuda()

        model.zero_grad()
        batch_p = model.predict_sequence(x, x_feats, x_len, x_chunk_len)
        batch_preds = []
        for p in batch_p:
            batch_preds += p

        batch_y = y.view(-1)
        # Focus on non-pad elemens
        idx = batch_y >= 0
        batch_y = batch_y[idx]
        label = batch_y.to('cpu').numpy()
        #print(len(batch_preds), label.shape)
        #exit()

        # Accumulate predictions
        preds += list(batch_preds)
        labels += list(label)

        # Get loss
        loss = model.neg_log_likelihood(x, x_feats, x_len, x_chunk_len, y)
        #print(loss.item())
        total_loss_dev += loss.item()
    return total_loss_dev / len(test_dataloader), labels, preds

def evaluate_sequences(model, test_dataloader, classes):
    model.eval()
    preds = []; labels = []
    for x, x_feats, x_len, x_chunk_len, y in test_dataloader:
        x = x.cuda()
        x_feats = x_feats.cuda()
        x_len = x_len.cuda()
        x_chunk_len = x_chunk_len.cuda()
        y = y.cuda()

        model.zero_grad()
        batch_p = model.predict_sequence(x, x_feats, x_len, x_chunk_len)
        batch_preds = []
        for p in batch_p:
            batch_preds += p

        batch_y = y.view(-1)
        # Focus on non-pad elemens
        idx = batch_y >= 0
        batch_y = batch_y[idx]
        label = batch_y.to('cpu').numpy()

        preds.append(list(batch_preds))
        labels.append(list(label))
    return labels, preds

def evaluate_emissions(model, test_dataloader, loss_fn, classes):
    model.eval()
    total_loss_dev = 0
    preds = []; labels = []
    for x, x_feats, x_len, x_chunk_len, y in test_dataloader:
        x = x.cuda()
        x_feats = x_feats.cuda()
        x_len = x_len.cuda()
        x_chunk_len = x_chunk_len.cuda()
        y = y.cuda()

        model.zero_grad()

        logits = model(x, x_feats, x_len, x_chunk_len)
        logits = logits.view(-1, len(classes) + 2)
        batch_y = y.view(-1)

        # Focus the loss on non-pad elemens
        idx = batch_y >= 0
        batch_y = batch_y[idx]
        logits = logits[idx]

        # Get predictions
        _, batch_preds = torch.max(logits, 1)
        pred = batch_preds.detach().cpu().numpy()
        label = batch_y.to('cpu').numpy()
        # Accumulate predictions

        preds += list(pred)
        labels += list(label)

        # Get loss
        loss = loss_fn(logits, batch_y)
        #print(loss.item())
        total_loss_dev += loss.item()
    return total_loss_dev / len(test_dataloader), labels, preds


def hot_start_emissions(args, model, train_dataloader, dev_dataloader, classes, class_weights):
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    # Training loop
    patience = 0; best_f1 = 0; epoch = 0; best_loss = 10000
    while True:
        #pbar = tqdm(total=len(train_dataloader))
        model.train()

        total_loss = 0

        for x, x_feats, x_len, x_chunk_len, y in train_dataloader:
            x = x.cuda()
            x_feats = x_feats.cuda()
            x_len = x_len.cuda()
            x_chunk_len = x_chunk_len.cuda()
            y = y.cuda()

            model.zero_grad()

            logits = model(x, x_feats, x_len, x_chunk_len)

            logits = logits.view(-1, len(classes) + 2)
            batch_y = y.view(-1)

            # Focus the loss on non-pad elemens
            idx = batch_y >= 0
            batch_y = batch_y[idx]
            logits = logits[idx]

            loss = loss_fn(logits, batch_y)
            #print(loss.item())
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            #pbar.update(1)

        dev_loss, dev_labels, dev_preds = evaluate_emissions(model, dev_dataloader, loss_fn, classes)
        macro_f1 = f1_score(dev_labels, dev_preds, average='macro')
        #if macro_f1 > best_f1:
        if dev_loss < best_loss:
            # Save model
            #print("Saving model...")
            torch.save(model.state_dict(), args.savedir_fold)
            best_f1 = macro_f1
            best_loss = dev_loss
            patience = 0
        else:
            patience += 1

        print("epoch {} loss {} dev_loss {} dev_macro_f1 {}".format(
            epoch,
            total_loss / len(train_dataloader),
            dev_loss,
            macro_f1))
        epoch += 1
        if patience >= args.patience:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hs_emissions', action='store_true')
    parser.add_argument('--use_transition_priors', action='store_true')
    parser.add_argument('--protocol', type=str,  help='protocol', required=True)
    parser.add_argument('--printout', default=False, action='store_true')
    parser.add_argument('--features', default=False, action='store_true')
    parser.add_argument('--token_level', default=False, action='store_true', help='perform prediction at token level')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--word_embed_path', type=str)
    parser.add_argument('--word_embed_size', type=int, default=100)
    parser.add_argument('--token_hidden_dim', type=int, default=50)
    parser.add_argument('--chunk_hidden_dim', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--do_train', default=False, action='store_true')
    parser.add_argument('--do_eval', default=False, action='store_true')
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--write_results', default=False, action='store_true')

    # I am not sure about what this is anymore
    parser.add_argument('--partition_sentence', default=False, action='store_true')
    args = parser.parse_args()

    protocols = ["TCP", "SCTP", "PPTP", "LTP", "DCCP", "BGPv4"]
    if args.protocol not in protocols:
        print("Specify a valid protocol")
        exit(-1)

    args.savedir_fold = os.path.join(args.savedir, "checkpoint_{}.pt".format(args.protocol))

    word2id = {}; tag2id = {}; pos2id = {}; id2cap = {}; stem2id = {}; id2word = {}
    transition_counts = {}
    # Get variable and state definitions
    def_vars = set(); def_states = set(); def_events = set(); def_events_constrained = set()
    data_utils.get_definitions(def_vars, def_states, def_events_constrained, def_events)

    together_path_list = [p for p in protocols if p != args.protocol]
    args.train = ["rfcs-bio/{}_phrases_train.txt".format(p) for p in together_path_list]
    args.test = ["rfcs-bio/{}_phrases.txt".format(args.protocol)]

    X_train_data_orig, y_train = data_utils.get_data(args.train, word2id, tag2id, pos2id, id2cap, stem2id, id2word, transition_counts, partition_sentence=args.partition_sentence)
    X_test_data_orig, y_test = data_utils.get_data(args.test, word2id, tag2id, pos2id, id2cap, stem2id, id2word, partition_sentence=args.partition_sentence)
    id2tag = {v: k for k, v in tag2id.items()}
    #print(id2tag)

    transition_priors = np.zeros((9, 9))
    for i in transition_counts:
        for j in transition_counts[i]:
            transition_priors[i][j] = transition_counts[i][j]

    transition_priors = normalize(transition_priors, axis=1, norm='l1')


    def_var_ids = [word2id[x.lower()] for x in def_vars if x.lower() in word2id]
    def_state_ids = [word2id[x.lower()] for x in def_states if x.lower() in word2id]
    def_event_ids = [word2id[x.lower()] for x in def_events if x.lower() in word2id] 

    max_chunks, max_tokens = data_utils.max_lengths(X_train_data_orig, y_train)
    max_chunks_test, max_tokens_test = data_utils.max_lengths(X_test_data_orig, y_test)

    max_chunks = max(max_chunks, max_chunks_test)
    max_tokens = max(max_tokens, max_tokens_test)

    print(max_chunks, max_tokens)
    #exit()

    id2tag = {v: k for k, v in tag2id.items()}

    vocab_size = len(stem2id)
    pos_size = len(pos2id)
    X_train_feats = features.transform_features(X_train_data_orig, vocab_size, pos_size, def_var_ids, def_state_ids, def_event_ids, id2cap, id2word, True)
    X_test_feats = features.transform_features(X_test_data_orig, vocab_size, pos_size, def_var_ids, def_state_ids, def_event_ids, id2cap, id2word, True)

    X_train_data, y_train, x_len, x_chunk_len =\
            data_utils.pad_sequences(X_train_data_orig, y_train, max_chunks, max_tokens)
    X_test_data, y_test, x_len_test, x_chunk_len_test =\
            data_utils.pad_sequences(X_test_data_orig, y_test, max_chunks, max_tokens)

    X_train_feats = data_utils.pad_features(X_train_feats, max_chunks)
    X_test_feats = data_utils.pad_features(X_test_feats, max_chunks)

    # Subsample a development set (10% of the data)
    n_dev = int(X_train_data.shape[0] * 0.1)
    dev_excerpt = random.sample(range(0, X_train_data.shape[0]), n_dev)
    train_excerpt = [i for i in range(0, X_train_data.shape[0]) if i not in dev_excerpt]

    X_dev_data = X_train_data[dev_excerpt]
    y_dev = y_train[dev_excerpt]
    x_len_dev = x_len[dev_excerpt]
    x_chunk_len_dev = x_chunk_len[dev_excerpt]
    X_dev_feats = X_train_feats[dev_excerpt]

    X_train_data = X_train_data[train_excerpt]
    y_train = y_train[train_excerpt]
    x_len = x_len[train_excerpt]
    x_chunk_len = x_chunk_len[train_excerpt]
    X_train_feats = X_train_feats[train_excerpt]

    #print(X_train_data.shape, y_train.shape, x_len.shape, x_chunk_len.shape)
    #print(X_dev_data.shape, y_dev.shape, x_len_dev.shape, x_chunk_len_dev.shape)
    print(X_train_data.shape, X_train_feats.shape, y_train.shape, x_len.shape, x_chunk_len.shape)
    print(X_dev_data.shape, X_dev_feats.shape, y_dev.shape, x_len_dev.shape, x_chunk_len_dev.shape)
    feat_sz = X_train_feats.shape[2]
    #print(x_chunk_len)
    #exit()

    y_train_ints = list(map(int, y_train.flatten()))
    y_train_ints = [y for y in y_train_ints if y >= 0]
    classes = list(set(y_train_ints))
    print(classes, tag2id)
    class_weights = list(compute_class_weight('balanced', classes, y_train_ints))
    class_weights += [0.0, 0.0]
    class_weights = torch.FloatTensor(class_weights).cuda()

    train_dataset = data_utils.ChunkDataset(X_train_data, X_train_feats, x_len, x_chunk_len, y_train)
    dev_dataset = data_utils.ChunkDataset(X_dev_data, X_dev_feats, x_len_dev, x_chunk_len_dev, y_dev)
    test_dataset = data_utils.ChunkDataset(X_test_data, X_test_feats, x_len_test, x_chunk_len_test, y_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load pre-trained embedding matrix
    #pretrained_emb = data_utils.load_glove_embeddings(args.word_embed_path, read_size=200000, embedding_dim=args.word_embed_size, word2id=word2id)
    pretrained_emb = data_utils.load_network_embeddings(args.word_embed_path, embedding_dim=args.word_embed_size, word2id=word2id)

    # Create model
    vocab_size = len(word2id)
    model = HierarchicalBiLSTM_CRF(vocab_size, args.word_embed_size, pretrained_emb,
                                args.token_hidden_dim, args.chunk_hidden_dim,
                                max_chunks, max_tokens, feat_sz, args.batch_size, output_dim=len(tag2id),
                                use_features=args.features)
    model.cuda()

    if args.do_train:
        if args.hs_emissions:
            hot_start_emissions(args, model, train_dataloader, dev_dataloader, classes, class_weights)
            model.load_state_dict(torch.load(args.savedir_fold, map_location=lambda storage, loc: storage))

        if args.use_transition_priors:
            model.load_transition_priors(transition_priors)

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

        # Training loop
        patience = 0; best_f1 = 0; epoch = 0; best_loss = 10000
        while True:
            #pbar = tqdm(total=len(train_dataloader))
            model.train()

            total_loss = 0

            for x, x_feats, x_len, x_chunk_len, y in train_dataloader:
                x = x.cuda()
                x_feats = x_feats.cuda()
                x_len = x_len.cuda()
                x_chunk_len = x_chunk_len.cuda()
                y = y.cuda()

                model.zero_grad()

                loss = model.neg_log_likelihood(x, x_feats, x_len, x_chunk_len, y)
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                #pbar.update(1)

            dev_loss, dev_labels, dev_preds = evaluate(model, dev_dataloader, classes)
            macro_f1 = f1_score(dev_labels, dev_preds, average='macro')
            #if macro_f1 > best_f1:
            if dev_loss < best_loss:
                # Save model
                #print("Saving model...")
                torch.save(model.state_dict(), args.savedir_fold)
                best_f1 = macro_f1
                best_loss = dev_loss
                patience = 0
            else:
                patience += 1

            print("epoch {} loss {} dev_loss {} dev_macro_f1 {}".format(
                epoch,
                total_loss / len(train_dataloader),
                dev_loss,
                macro_f1))
            epoch += 1
            if patience >= args.patience:
                break

    if args.do_eval:
        # Load model
        model.load_state_dict(torch.load(args.savedir_fold, map_location=lambda storage, loc: storage))

        y_test, y_pred = evaluate_sequences(model, test_dataloader, classes)
        
        y_test_trans = data_utils.translate(y_test, id2tag)
        y_pred_trans = data_utils.translate(y_pred, id2tag)
        #print(y_test_trans)
        _, y_test_trans = data_utils.expand(X_test_data_orig, y_test_trans)
        X_test_data_orig, y_pred_trans = data_utils.expand(X_test_data_orig, y_pred_trans)
        y_test_flatten = data_utils.flatten(y_test_trans)
        y_pred_flatten = data_utils.flatten(y_pred_trans)

        print(classification_report(y_test_flatten, y_pred_flatten))
        print("ACC", accuracy_score(y_test_flatten, y_pred_flatten))
        print("WEIGHTED f1", f1_score(y_test_flatten, y_pred_flatten, average='weighted'))
        print("MACRO F1", f1_score(y_test_flatten, y_pred_flatten, average='macro'))

        evaluator = Evaluator(y_test_trans, y_pred_trans,
                              ['ACTION', 'ERROR', 'TIMER', 'TRANSITION', 'TRIGGER', 'VARIABLE'])
        results, results_agg = evaluator.evaluate()

        for measure in results:
            precision = results[measure]['precision']
            recall = results[measure]['precision']
            if (precision + recall) <= 0:
                f1 = 0.0
            else:
                f1 = 2.0 * ((precision * recall) / (precision + recall))
            print(measure, f1)
        for tag in ['TRIGGER', 'ACTION', 'ERROR', 'TIMER', 'TRANSITION', 'VARIABLE']:
            evaluator = Evaluator(y_test_trans, y_pred_trans, [tag])
            results, results_agg = evaluator.evaluate()
            for measure in results:
                precision = results[measure]['precision']
                recall = results[measure]['recall']
                if (precision + recall) <= 0:
                    f1 = 0.0
                else:
                    f1 = 2.0 * ((precision * recall) / (precision + recall))
                print(tag, measure, f1)

        if args.write_results:
            def_states_protocol = {}; def_events_protocol = {}; def_events_constrained_protocol = {}
            data_utils.get_protocol_definitions(args.protocol, def_states_protocol, def_events_constrained_protocol, def_events_protocol)

            # Trying a heuristic
            #y_pred_trans = data_utils.flip_outside_trigger(X_test_data_orig, y_test_trans, y_pred_trans, id2word, def_states_protocol, def_events_protocol)
            #y_pred_trans = data_utils.flip_trigger_mistakes(X_test_data_orig, y_test_trans, y_pred_trans, id2word, def_states_protocol)
            y_pred_trans = data_utils.heuristic_transitions(X_test_data_orig, y_test_trans, y_pred_trans, id2word, def_states_protocol)
            y_pred_trans = data_utils.heuristic_actions(X_test_data_orig, y_test_trans, y_pred_trans, id2word, def_events_protocol)
            y_pred_trans = data_utils.heuristic_outside(X_test_data_orig, y_test_trans, y_pred_trans, id2word, def_states_protocol, def_events_protocol)
            y_pred_trans = data_utils.join_consecutive_transitions(X_test_data_orig, y_test_trans, y_pred_trans)

            output_xml = os.path.join(args.outdir, "{}.xml".format(args.protocol))
            results = data_utils.write_results(X_test_data_orig, y_test_trans, y_pred_trans, id2word, def_states_protocol, def_events_protocol, def_events_constrained_protocol)
            with open(output_xml, "w") as fp:
                fp.write(results)

if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(4321)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(4321)
    random.seed(4321)

    main()
