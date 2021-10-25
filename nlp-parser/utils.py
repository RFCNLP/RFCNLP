from ner_evaluation.ner_eval import Evaluator
import data_utils_NEW as data_utils

def evaluate(y_test_trans_eval, y_pred_trans_eval):
    y_test_flatten = data_utils.flatten(y_test_trans_eval)
    y_pred_flatten = data_utils.flatten(y_pred_trans_eval)

    print(classification_report(y_test_flatten, y_pred_flatten), digits=4)
    print("ACC", accuracy_score(y_test_flatten, y_pred_flatten), digits=4)
    print("WEIGHTED f1", f1_score(y_test_flatten, y_pred_flatten, average='weighted'), digits=4)
    print("MACRO F1", f1_score(y_test_flatten, y_pred_flatten, average='macro'), digits=4)


    evaluator = Evaluator(y_test_trans_eval, y_pred_trans_eval,
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
        evaluator = Evaluator(y_test_trans_eval, y_pred_trans_eval, [tag])
        results, results_agg = evaluator.evaluate()
        for measure in results:
            precision = results[measure]['precision']
            recall = results[measure]['recall']
            if (precision + recall) <= 0:
                f1 = 0.0
            else:
                f1 = 2.0 * ((precision * recall) / (precision + recall))
            print(tag, measure, f1)

def apply_heuristics(X_test_data_alt, y_test_trans_alt, y_pred_trans_alt,
                    id2word, def_states_protocol, def_events_protocol,
                    transitions=False, outside=False, actions=False,
                    consecutive_trans=False):
    if transitions:
        y_pred_trans_alt = data_utils.heuristic_transitions(X_test_data_alt, y_test_trans_alt, y_pred_trans_alt, id2word, def_states_protocol)
    if outside:
        y_pred_trans_alt = data_utils.heuristic_outside(X_test_data_alt, y_test_trans_alt, y_pred_trans_alt, id2word, def_states_protocol, def_events_protocol)
    if actions:
        y_pred_trans_alt = data_utils.heuristic_actions(X_test_data_alt, y_test_trans_alt, y_pred_trans_alt, id2word, def_events_protocol)
    if consecutive_trans:
        y_pred_trans_alt = data_utils.join_consecutive_transitions(X_test_data_alt, y_test_trans_alt, y_pred_trans_alt, level_h_alt, level_d_alt)

