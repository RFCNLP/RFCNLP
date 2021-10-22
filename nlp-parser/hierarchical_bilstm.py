import os
import torch.nn as nn
import torch
import numpy as np
import random
import argparse
from tqdm import tqdm, trange
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import *
from ner_evaluation.ner_eval import Evaluator

import data_utils
import features

class HierarchicalBiLSTM(nn.Module):

    def __init__(self, vocab_size, token_embedding_dim, pretrained,
                 token_hidden_dim, chunk_hidden_dim, max_chunk_len, max_seq_len,
                 feat_sz, batch_size, output_dim, use_features=False):
        super(HierarchicalBiLSTM, self).__init__()
        self.token_hidden_dim = token_hidden_dim
        self.chunk_hidden_dim = chunk_hidden_dim
        self.max_chunk_len = max_chunk_len
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

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

        self.fc = nn.Linear(2 * chunk_hidden_dim, output_dim)

        self.hidden_token_bilstm = self.init_hidden_token_bilstm()
        self.hidden_chunk_bilstm = self.init_hidden_chunk_bilstm()

    def init_hidden_token_bilstm(self):
        var1 = torch.autograd.Variable(torch.zeros(2, self.batch_size * self.max_chunk_len, self.token_hidden_dim)).cuda()
        var2 = torch.autograd.Variable(torch.zeros(2, self.batch_size * self.max_chunk_len, self.token_hidden_dim)).cuda()
        return var1, var2

    def init_hidden_chunk_bilstm(self):
        var1 = torch.autograd.Variable(torch.zeros(2, self.batch_size, self.chunk_hidden_dim)).cuda()
        var2 = torch.autograd.Variable(torch.zeros(2, self.batch_size, self.chunk_hidden_dim)).cuda()
        return var1, var2

    def forward(self, x, x_feats, x_len, x_chunk_len):
        self.batch_size = x.shape[0]

        self.hidden_token_bilstm = self.init_hidden_token_bilstm()
        self.hidden_chunk_bilstm = self.init_hidden_chunk_bilstm()

        x = self.embedding(x)
        # unsqueeze all the chunks
        x = x.view(self.batch_size * self.max_chunk_len, self.max_seq_len, -1)
        # clamp everything to minimum length of 1, but keep the original variable to mask the output later
        x_chunk_len = x_chunk_len.view(-1)
        x_chunk_len_clamped = x_chunk_len.clamp(min=1, max=self.max_seq_len)

        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_chunk_len_clamped, batch_first=True, enforce_sorted=False)
        x, self.hidden_token_bilstm = self.token_lstm(x, self.hidden_token_bilstm)
        # unpack
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # extract last timestep, since doing [-1] would get the padded zeros
        idx = (x_chunk_len_clamped - 1).view(-1, 1).expand(x.size(0), x.size(2)).unsqueeze(1)
        x = x.gather(1, idx).squeeze()

        # revert back to (batch_size, max_chunk_len)
        x = x.view(self.batch_size, self.max_chunk_len, -1)
        if self.use_features:
            x = torch.cat((x, x_feats), 2)

        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x, self.hidden_chunk_bilstm = self.chunk_lstm(x, self.hidden_chunk_bilstm)
        # unpack
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=self.max_chunk_len)
        x = self.fc(x)
        return x

def evaluate(model, test_dataloader, loss_fn, classes):
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
        logits = logits.view(-1, len(classes))
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

        logits = model(x, x_feats, x_len, x_chunk_len)
        logits = logits.view(-1, len(classes))
        batch_y = y.view(-1)

        # Focus the loss on non-pad elemens
        idx = batch_y >= 0
        batch_y = batch_y[idx]
        logits = logits[idx]

        # Get predictions
        _, batch_preds = torch.max(logits, 1)
        pred = batch_preds.detach().cpu().numpy()
        label = batch_y.to('cpu').numpy()

        preds.append(list(pred))
        labels.append(list(label))
    return labels, preds

def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--partition_sentence', default=False, action='store_true')
    args = parser.parse_args()

    protocols = ["TCP", "SCTP", "PPTP", "LTP", "DCCP", "BGPv4"]
    if args.protocol not in protocols:
        print("Specify a valid protocol")
        exit(-1)

    args.savedir_fold = os.path.join(args.savedir, "checkpoint_{}.pt".format(args.protocol))

    word2id = {}; tag2id = {}; pos2id = {}; id2cap = {}; stem2id = {}; id2word = {}
    # Get variable and state definitions
    def_vars = set(); def_states = set()
    data_utils.get_definitions(def_vars, def_states)

    together_path_list = [p for p in protocols if p != args.protocol]
    args.train = ["rfcs-bio/{}_phrases_train.txt".format(p) for p in together_path_list]
    args.test = ["rfcs-bio/{}_phrases.txt".format(args.protocol)]

    X_train_data_orig, y_train = data_utils.get_data(args.train, word2id, tag2id, pos2id, id2cap, stem2id, id2word, partition_sentence=args.partition_sentence)
    X_test_data_orig, y_test = data_utils.get_data(args.test, word2id, tag2id, pos2id, id2cap, stem2id, id2word, partition_sentence=args.partition_sentence)
    id2tag = {v: k for k, v in tag2id.items()}
    #print(id2tag)

    def_var_ids = [word2id[x.lower()] for x in def_vars if x.lower() in word2id]
    def_state_ids = [word2id[x.lower()] for x in def_states if x.lower() in word2id]

    max_chunks, max_tokens = data_utils.max_lengths(X_train_data_orig, y_train)
    max_chunks_test, max_tokens_test = data_utils.max_lengths(X_test_data_orig, y_test)

    max_chunks = max(max_chunks, max_chunks_test)
    max_tokens = max(max_tokens, max_tokens_test)

    print(max_chunks, max_tokens)
    #exit()

    vocab_size = len(stem2id)
    pos_size = len(pos2id)
    X_train_feats = features.transform_features(X_train_data_orig, vocab_size, pos_size, def_var_ids, def_state_ids, id2cap, id2word, True)
    X_test_feats = features.transform_features(X_test_data_orig, vocab_size, pos_size, def_var_ids, def_state_ids, id2cap, id2word, True)

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
    class_weights = torch.FloatTensor(compute_class_weight('balanced', classes, y_train_ints)).cuda()
    print(class_weights)

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
    model = HierarchicalBiLSTM(vocab_size, args.word_embed_size, pretrained_emb,
                                args.token_hidden_dim, args.chunk_hidden_dim,
                                max_chunks, max_tokens, feat_sz, args.batch_size, output_dim=len(classes),
                                use_features=args.features)
    model.cuda()

    if args.do_train:
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

                logits = logits.view(-1, len(classes))
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

            dev_loss, dev_labels, dev_preds = evaluate(model, dev_dataloader, loss_fn, classes)
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
        _, y_pred_trans = data_utils.expand(X_test_data_orig, y_pred_trans)
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


if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(4321)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(4321)
    random.seed(4321)

    main()
