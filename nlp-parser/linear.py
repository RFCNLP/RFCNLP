from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM
import numpy as np
import argparse
import re
import os

import features
import data_utils_NEW as data_utils
from eval_utils import evaluate, apply_heuristics

def error_analysis(X_test_data, y_test_trans, y_pred_trans, id2word):
    for x, y_g, y_p in zip(X_test_data, y_test_trans, y_pred_trans):
        x_trans = [id2word[i] for i in x]
        for word, pred_tag, true_tag in zip(x_trans, y_p, y_g):
            print("{0}\t{1}\t|\t{2}".format(word, pred_tag, true_tag))
        print("=======")
        print()

def main():
    np.random.seed(4321)
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=str,  help='protocol', required=True)
    parser.add_argument('--printout', default=False, action='store_true')
    parser.add_argument('--stem', default=False, action='store_true')
    parser.add_argument('--write_results', default=False, action='store_true')
    parser.add_argument('--heuristics', default=False, action='store_true')
    parser.add_argument('--heuristics_only', default=False, action='store_true')
    parser.add_argument('--token_level', default=False, action='store_true', help='perform prediction at token level')
    parser.add_argument('--phrase_level', default=False, action='store_true', help='perform prediction at the phrase level')
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()

    protocols = ["TCP", "SCTP", "PPTP", "LTP", "DCCP", "BGPv4"]
    if args.protocol not in protocols:
        print("Specify a valid protocol")
        exit(-1)

    word2id = {}; tag2id = {}; pos2id = {}; id2cap = {}; stem2id = {}; id2word = {}
    # Get variable, state and event definitions
    def_vars = set(); def_states = set(); def_events_constrained = set(); def_events = set()
    data_utils.get_definitions(def_vars, def_states, def_events_constrained, def_events)


    together_path_list = [p for p in protocols if p != args.protocol]

    if args.token_level:
        args.train = ["rfcs-bio/{}_no_chunk_train.txt".format(p) for p in together_path_list]
        args.test = ["rfcs-bio/{}_no_chunk.txt".format(args.protocol)]
        X_train_data, y_train = data_utils.get_data_nochunks(args.train, word2id, tag2id, pos2id, id2cap, stem2id, id2word)
        X_test_data, y_test = data_utils.get_data_nochunks(args.test, word2id, tag2id, pos2id, id2cap, stem2id, id2word)
    elif args.phrase_level:
        args.train = ["rfcs-bio/{}_phrases_train.txt".format(p) for p in together_path_list]
        args.test = ["rfcs-bio/{}_phrases.txt".format(args.protocol)]
        X_train_data, y_train, level_h, level_d = data_utils.get_data(args.train, word2id, tag2id, pos2id, id2cap, stem2id)
        X_test_data, y_test, level_h, level_d = data_utils.get_data(args.test, word2id, tag2id, pos2id, id2cap, stem2id)
        #print(len(X_train_data), y_train.shape)
        #print(len(X_test_data), y_test.shape)
        #exit()
    else:
        args.train = ["rfcs-bio/{}_train.txt".format(p) for p in together_path_list]
        args.test = ["rfcs-bio/{}.txt".format(args.protocol)]
        X_train_data, y_train, level_h, level_d = data_utils.get_data(args.train, word2id, tag2id, pos2id, id2cap, stem2id)
        X_test_data, y_test, level_h, level_d = data_utils.get_data(args.test, word2id, tag2id, pos2id, id2cap, stem2id)

    if args.stem:
        vocab_size = len(stem2id)
    else:
        vocab_size = len(word2id)
    pos_size = len(pos2id)

    #error_id = word2id['error']
    #timer_ids = [word2id['timer'], word2id['timers']]
    def_var_ids = [word2id[x.lower()] for x in def_vars if x.lower() in word2id]
    def_state_ids = [word2id[x.lower()] for x in def_states if x.lower() in word2id]
    def_event_ids = [word2id[x.lower()] for x in def_events if x.lower() in word2id] 

    #print(def_var_ids)
    #print(def_state_ids)
    #exit()
    print(tag2id)
    id2tag = {v: k for k, v in tag2id.items()}
    id2word = {v: k for k, v in word2id.items()}

    if not args.token_level:
        X_train = features.transform_features(X_train_data, vocab_size, pos_size, def_var_ids, def_state_ids, def_event_ids, id2cap, id2word, word2id, args.stem)
        X_test = features.transform_features(X_test_data, vocab_size, pos_size, def_var_ids, def_state_ids, def_event_ids, id2cap, id2word, word2id, args.stem)
    else:
        X_train = features.transform_features_nochunk(X_train_data, vocab_size, pos_size, def_var_ids, def_state_ids, def_event_ids, id2cap, id2word, word2id, args.stem)
        X_test = features.transform_features_nochunk(X_test_data, vocab_size, pos_size, def_var_ids, def_state_ids, def_event_ids, id2cap, id2word, word2id, args.stem)

    #exit()
    print("vocabulary size", vocab_size)

    model = ChainCRF()
    ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=10, verbose=1, show_loss_every=1)
    ssvm.fit(X_train, y_train)

    y_pred = ssvm.predict(X_test)

    y_test_trans = data_utils.translate(y_test, id2tag)
    y_pred_trans = data_utils.translate(y_pred, id2tag)

    if not args.token_level:
        # Do it in a way that preserves the original chunk-level segmentation
        _, y_test_trans_alt, _, _ = data_utils.alternative_expand(X_test_data, y_test_trans, level_h, level_d, id2word, debug=False)
        X_test_data_alt, y_pred_trans_alt, level_h_alt, level_d_alt = data_utils.alternative_expand(X_test_data, y_pred_trans, level_h, level_d, id2word, debug=True)

        # Do it in a way that flattens the chunk-level segmentation for evaluation
        X_test_data_old = X_test_data[:]
        _, y_test_trans_eval = data_utils.expand(X_test_data, y_test_trans, id2word, debug=False)
        X_test_data_eval, y_pred_trans_eval = data_utils.expand(X_test_data, y_pred_trans, id2word, debug=True)
    else:
        y_test_trans_eval = y_test_trans
        y_pred_trans_eval = y_pred_trans
        X_test_data_eval = X_test_data


    #print(len(y_pred_trans), len(y_test_trans))
    #exit()
    evaluate(y_test_trans_eval, y_pred_trans_eval)

    def_states_protocol = {}; def_events_protocol = {}; def_events_constrained_protocol = {}; def_variables_protocol = {}
    data_utils.get_protocol_definitions(args.protocol, def_states_protocol, def_events_constrained_protocol, def_events_protocol, def_variables_protocol)

    # Trying a heuristic
    if args.heuristics_only:
        for i in range(0, len(y_pred_trans_alt)):
            for j in range(0, len(y_pred_trans_alt[i])):
                for k in range(0, len(y_pred_trans_alt[i][j])):
                    y_pred_trans_alt[i][j][k] = 'O'

    y_pred_trans_alt =\
        apply_heuristics(X_test_data_alt, y_test_trans_alt, y_pred_trans_alt,
                         level_h_alt, level_d_alt,
                         id2word, def_states_protocol, def_events_protocol, def_variables_protocol,
                         transitions=args.heuristics, outside=args.heuristics, actions=args.heuristics)
    y_pred_trans_alt = \
        apply_heuristics(X_test_data_alt, y_test_trans_alt, y_pred_trans_alt,
                     level_h_alt, level_d_alt,
                     id2word, def_states_protocol, def_events_protocol, def_variables_protocol,
                     consecutive_trans=True)

    X_test_data, y_pred_trans, level_h_trans, level_d_trans = \
        data_utils.alternative_join(
                X_test_data_alt, y_pred_trans_alt,
                level_h_alt, level_d_alt,
                id2word, debug=True)

    # Do it in a way that flattens the chunk-level segmentation for evaluation
    _, y_test_trans_eval = data_utils.expand(X_test_data_old, y_test_trans, id2word, debug=False)
    #X_test_data_eval, y_pred_trans_eval = data_utils.expand(X_test_data, y_pred_trans, id2word, debug=True)

    if args.heuristics:
        evaluate(y_test_trans_eval, y_pred_trans)

    if args.write_results:
        output_xml = os.path.join(args.outdir, "{}.xml".format(args.protocol))
        results = data_utils.write_results(X_test_data, y_test_trans, y_pred_trans, level_h_trans, level_d_trans,
                                          id2word, def_states_protocol, def_events_protocol, def_events_constrained_protocol,
                                          args.protocol, cuda_device=-1)
        with open(output_xml, "w") as fp:
            fp.write(results)


if __name__ == "__main__":
    main()

