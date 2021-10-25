import re
import numpy as np

def chunk_features(chunk_w, chunk_pos, chunk_stem, vocab_size, pos_size, variable_ids, state_ids, event_ids, id2cap, id2word, word2id, n_chunk_pos, n_chunk_len, use_stem):
    vector = [0] * vocab_size
    vector_pos = [0] * pos_size
    vector_cap_pattern = [0] * 7
    n_chunk_words = len(chunk_w)

    has_error = 0; has_timer = 0; has_variable = 0; has_state = 0; has_event = 0; has_state_regex = 0;
    has_timer_verbs = 0; has_variable_verbs = 0; has_transition_verbs = 0; has_action_verbs = 0;
    has_transition_dir = 0; has_conditional_words = 0

    # Add other features to capture variables
    assignment_regex = r'(^\w*(<-|:=|=)\w*$)|(set|sets|.*increment.*)'
    assignment_logic = 0;

    comparison_regex = r'.*(==|<|>|<=|>=).*'
    comparison_logic = 1;

    math_regex = r'^([a-z_\.]+(\+|-|\*)\d+)$|(^(max|min)$)'
    math_logic = 1;

    timer_regex = r'.*timer.*'
    error_regex = r'.*error.*'
    state_regex = r'.*state.*'

    timer_verbs = r'start.*|restart.*|stop.*|reset.*'
    variable_verbs = r'delet.*|set.*|increment.*'
    action_verbs = r'send.*|receiv.*|issu.*|generat.*|form.*|creat.*'
    transition_verbs = r'enter.*|mov.*|chang.*|stay.*|leav.*|go.*|remain*|.*\w\.state :=.*'

    conditional_words = r'if|when|otherwise|while'
    transition_dir = r'to|from|through'

    for (token, pos, stem) in zip(chunk_w, chunk_pos, chunk_stem):
        #print(id2word[token])
        # in the BERT case we are keeping cases, but these features are lowercased
        token_prev = token
        token = word2id[id2word[token].lower()]

        if re.match(assignment_regex, id2word[token].lower()):
            assignment_logic = 1
            #print(id2word[token])
        if re.match(comparison_regex, id2word[token].lower()):
            comparison_logic = 1
        if re.match(math_regex, id2word[token].lower()):
            math_logic = 1
        if re.match(timer_regex, id2word[token].lower()):
            has_timer = 1
        if re.match(timer_verbs, id2word[token].lower()):
            has_timer_verbs = 1
        if re.match(variable_verbs, id2word[token].lower()):
            has_variable_verbs = 1
        if re.match(error_regex, id2word[token].lower()):
            has_error = 1
        if token in variable_ids:
            has_variable = 1
        elif token in state_ids:
            has_state = 1
        elif token in event_ids:
            has_event = 1
        elif re.match(state_regex, id2word[token].lower()):
            has_state_regex = 1
        elif re.match(transition_verbs, id2word[token].lower()):
            has_transition_verbs = 1
        elif re.match(transition_dir, id2word[token].lower()):
            has_transition_dir = 1
        elif re.match(action_verbs, id2word[token].lower()):
            has_action_verbs = 1
        elif re.match(conditional_words, id2word[token].lower()):
            has_conditional_words = 1

        if use_stem:
            vector[stem] = 1
        else:
            vector[token] = 1
        vector_pos[pos] = 1
        vector_cap_pattern[id2cap[token_prev]] = 1

    return vector + vector_pos + vector_cap_pattern + \
    [
    has_error, has_timer, has_variable, 
    has_state, has_event,
    has_state_regex, has_transition_verbs, has_transition_dir,
    has_timer_verbs, has_variable_verbs,
    has_action_verbs,
    assignment_logic, comparison_logic, math_logic,
    has_conditional_words, n_chunk_words,
    (1.0 * n_chunk_pos)/n_chunk_len
    ]

def token_features(token, pos, stem, vocab_size, pos_size, variable_ids, state_ids, event_ids, id2cap, id2word, word2id, n_token_pos, n_token_len, use_stem):
    vector = [0] * vocab_size
    vector_pos = [0] * pos_size
    vector_cap_pattern = [0] * 7

    has_error = 0; has_timer = 0; has_variable = 0; has_state = 0; has_event = 0; has_state_regex = 0;
    has_timer_verbs = 0; has_variable_verbs = 0; has_transition_verbs = 0; has_action_verbs = 0
    has_transition_dir = 0; has_conditional_words = 0

    # Add other features to capture variables
    assignment_regex = r'(^\w*(<-|:=|=)\w*$)|(set|sets|.*increment.*|.*increments.*)'
    assignment_logic = 0;

    comparison_regex = r'.*(==|<|>|<=|>=).*'
    comparison_logic = 1;

    math_regex = r'^([a-z_\.]+(\+|-|\*)\d+)$|(^(max|min)$)'
    math_logic = 1;

    timer_regex = r'.*timer.*'
    error_regex = r'.*error.*'
    state_regex = r'.*state.*'

    timer_verbs = r'start.*|restart.*|stop.*|reset.*'
    variable_verbs = r'delet.*|set.*|increment.*'
    action_verbs = r'send.*|receiv.*|issu.*|generat.*|form.*|creat.*'
    transition_verbs = r'enter.*|mov.*|chang.*|stay.*|leav.*|go.*|remain*|.*\w\.state :=.*'

    conditional_words = r'if|when|otherwise|while'
    transition_dir = r'to|from|through'

    if token is not None:
        prev_token = token
        token = word2id[id2word[token].lower()]
        
        if re.match(assignment_regex, id2word[token].lower()):
            assignment_logic = 1
            #print(id2word[token])
        if re.match(comparison_regex, id2word[token].lower()):
            comparison_logic = 1
        if re.match(math_regex, id2word[token].lower()):
            math_logic = 1
        if re.match(timer_regex, id2word[token].lower()):
            has_timer = 1
        if re.match(timer_verbs, id2word[token].lower()):
            has_timer_verbs = 1
        if re.match(variable_verbs, id2word[token].lower()):
            has_variable_verbs = 1
        if re.match(error_regex, id2word[token].lower()):
            has_error = 1
        if token in variable_ids:
            has_variable = 1
        elif token in state_ids:
            has_state = 1
        elif token in event_ids:
            has_event = 1
        elif re.match(state_regex, id2word[token].lower()):
            has_state_regex = 1
        elif re.match(transition_verbs, id2word[token].lower()):
            has_transition_verbs = 1
        elif re.match(transition_dir, id2word[token].lower()):
            has_transition_dir = 1
        elif re.match(action_verbs, id2word[token].lower()):
            has_action_verbs = 1
        elif re.match(conditional_words, id2word[token].lower()):
            has_conditional_words = 1

        if use_stem:
            vector[stem] = 1
        else:
            vector[token] = 1
        vector_pos[pos] = 1
        vector_cap_pattern[id2cap[prev_token]] = 1

    return vector + vector_pos + vector_cap_pattern + \
    [
    has_error, has_timer, has_variable, 
    has_state, has_event,
    has_state_regex, has_transition_verbs, has_transition_dir,
    has_timer_verbs, has_variable_verbs,
    has_action_verbs,
    assignment_logic, comparison_logic, math_logic,
    has_conditional_words,
    (1.0 * n_token_pos)/n_token_len]


def transform_features(X, vocab_size, pos_size, variable_ids, state_ids, event_ids, id2cap, id2word={}, word2id={}, use_stem=False):
    X_new = []
    for i, sequence in enumerate(X):
        x_sequence = []

        n_chunk_len = len(sequence)
        for n_chunk_pos, (chunk_w, chunk_pos, chunk_stem) in enumerate(sequence):
            # Get previous chunk
            two_previous_tokens = []; two_previous_pos = []; two_previous_stem = []
            previous_tokens = []; previous_pos = []; previous_stem = []
            if n_chunk_pos - 1 > 0:
                previous_tokens = sequence[n_chunk_pos - 1][0]
                previous_pos = sequence[n_chunk_pos - 1][1]
                previous_stem = sequence[n_chunk_pos - 1][2]
            if n_chunk_pos - 2 > 0:
                two_previous_tokens = sequence[n_chunk_pos - 2][0]
                two_previous_pos = sequence[n_chunk_pos - 2][1]
                two_previous_stem = sequence[n_chunk_pos - 2][2]

            # Get features for the currenta and the previous chunk
            curr_chunk_feats = chunk_features(chunk_w, chunk_pos, chunk_stem, vocab_size, pos_size, variable_ids, state_ids, event_ids, id2cap, id2word, word2id, n_chunk_pos + 1, n_chunk_len, use_stem)
            prev_chunk_feats = chunk_features(previous_tokens, previous_pos, previous_stem, vocab_size, pos_size, variable_ids, state_ids, event_ids, id2cap, id2word, word2id, n_chunk_pos, n_chunk_len, use_stem)
            two_prev_chunk_feats = chunk_features(two_previous_tokens, two_previous_pos, two_previous_stem, vocab_size, pos_size, variable_ids, state_ids, event_ids, id2cap, id2word, word2id, n_chunk_pos - 1, n_chunk_len, use_stem)

            #print(curr_chunk_feats)
            #print('----')
            #print(prev_chunk_feats)

            #x_sequence.append(curr_chunk_feats + prev_chunk_feats)
            x_sequence.append(curr_chunk_feats + prev_chunk_feats + two_prev_chunk_feats)

        X_new.append(np.array(x_sequence))
    #exit()
    return np.array(X_new)

def transform_features_nochunk(X, vocab_size, pos_size, variable_ids, state_ids, event_ids, id2cap, id2word={}, word2id={}, use_stem=False):
    X_new = []
    for i, sequence in enumerate(X):
        x_sequence = []

        n_seq_len = len(sequence[0])

        for n_token_pos, (token, pos, stem) in enumerate(zip(sequence[0], sequence[1], sequence[2])):
            # Get previous chunk
            previous_token = None; previous_pos = None; previous_stem = None
            if n_token_pos - 1 > 0:
                previous_token = sequence[0][n_token_pos - 1]
                previous_pos = sequence[1][n_token_pos - 1]
                previous_stem = sequence[2][n_token_pos - 1]

            # Get features for the currenta and the previous chunk
            curr_token_feats = token_features(token, pos, stem, vocab_size, pos_size, variable_ids, state_ids, event_ids, id2cap, id2word, word2id, n_token_pos + 1, n_seq_len, use_stem)
            prev_token_feats = token_features(previous_token, previous_pos, previous_stem, vocab_size, pos_size, variable_ids, state_ids, event_ids, id2cap, id2word, word2id, n_token_pos, n_seq_len, use_stem)

            x_sequence.append(curr_token_feats + prev_token_feats)

        X_new.append(np.array(x_sequence))
    #exit()
    return np.array(X_new)

