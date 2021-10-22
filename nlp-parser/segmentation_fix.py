from collections import Counter
import re
import argparse

def get_data(filename):
    words = []; tags = []; pos = []; relevant = []
    with open(filename) as fp:
        for line in fp:
            elems = line.strip().split(' ======= ')
            if elems[0] in ['END-OF-CHUNK', 'END-OF-CONTROL']:
                words.append(elems[0])
                tags.append(elems[1])
                pos.append(None)
                relevant.append(None)
            else:
                words.append(elems[0])
                tags.append(elems[1])
                pos.append(elems[2])
                relevant.append(elems[3])
    return words, tags, pos, relevant

def write_segmentation_fix(filename, words, tags, pos, relevant):
    with open(filename, "w") as fp:
        for w, t, p, r in zip(words, tags, pos, relevant):
            if p and r:
                fp.write("{0} ======= {1} ======= {2} ======= {3}\n".format(w, t, p, r))
            else:
                fp.write("{0} ======= {1}\n".format(w, t))


def rechunk_quotations(words, tags, pos, relevant):
    new_words = []; new_tags = []; new_pos = []; new_relevant = []

    curr_tags = []; curr_words = []; curr_pos = []; curr_relevant = []
    first_quote = False; last_quote = False
    for w, t, p, r in zip(words, tags, pos, relevant):
        if w == '``' and not first_quote:
            curr_prev_w = prev_w
            curr_words.append(w)
            curr_tags.append(t)
            curr_pos.append(p)
            curr_relevant.append(r)
            first_quote = True
        elif w == '``' and first_quote:
            curr_words.append(w)
            curr_tags.append(t)
            curr_pos.append(p)
            curr_relevant.append(r)
            #print("PREV--", curr_prev_w)
            #print(" ".join(curr_words))
            #print(" ".join(curr_tags))
            #print(curr_tags)
            #print('=====')

            data = Counter(curr_tags)
            chunk_tag = data.most_common(1)[0][0]
            begin_chunk_tag = re.sub('I-', 'B-', chunk_tag)
            inside_chunk_tag = re.sub('B-', 'I-', chunk_tag)

            first_end_of_chunk = 0
            if 'END-OF-CHUNK' in curr_words:
                first_end_of_chunk = curr_words.index('END-OF-CHUNK')

            #print(curr_words)
            #print(curr_tags)
            #print(first_end_of_chunk, curr_tags[first_end_of_chunk])
            #print('----')
            #exit()

            first_tag = chunk_tag
            if chunk_tag.startswith('I'):
                if curr_tags[0] == begin_chunk_tag:
                    first_tag = begin_chunk_tag
                    chunk_tag = begin_chunk_tag
                elif curr_tags[first_end_of_chunk] == begin_chunk_tag:
                    first_tag = chunk_tag
                    chunk_tag = begin_chunk_tag

            index = 0
            for w_c, t_c, p_c, r_c in zip(curr_words, curr_tags, curr_pos, curr_relevant):
                if index == 0 and  w_c != 'END-OF-CHUNK':
                    new_words.append(w_c)
                    new_tags.append(first_tag)
                    new_pos.append(p_c)
                    new_relevant.append(r_c)
                elif w_c != 'END-OF-CHUNK':
                    new_words.append(w_c)
                    new_tags.append(inside_chunk_tag)
                    new_pos.append(p_c)
                    new_relevant.append(r_c)
                index += 1
            new_words.append('END-OF-CHUNK')
            new_tags.append(chunk_tag)
            new_pos.append(None)
            new_relevant.append(None)

            curr_tags = []; curr_words = []; curr_pos = []; curr_relevant = []
            first_quote = False; last_quote = True
        elif first_quote:
            curr_words.append(w)
            curr_tags.append(t)
            curr_pos.append(p)
            curr_relevant.append(r)
        elif last_quote and w == 'END-OF-CHUNK':
            last_quote = False
            continue
        else:
            new_words.append(w)
            new_tags.append(t)
            new_pos.append(p)
            new_relevant.append(r)
            last_quote = False

        prev_w = w
    return new_words, new_tags, new_pos, new_relevant


def sentence_split(words, tags, pos, relevant):
    control_str = []
    for w, t, p, r in zip(words, tags, pos, relevant):
        if w == "END-OF-CONTROL":
            print(control_str)
            control_str = []
        elif w != "END-OF-CHUNK":
            control_str.append(w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol_file', type=str,  help='protocol_file', required=True)
    args = parser.parse_args()
    #protocols = ["TCP", "SCTP", "PPTP", "LTP", "DCCP", "BGPv4"]
    #for protocol in protocols:
    words, tags, pos, relevant = get_data(args.protocol_file)
    #sentence_split(words, tags, pos, relevant)
    #exit()
    new_words, new_tags, new_pos, new_relevant = rechunk_quotations(words, tags, pos, relevant)
    write_segmentation_fix("{}_fixed.txt".format(args.protocol_file[0:-4]), new_words, new_tags, new_pos, new_relevant)


if __name__ == "__main__":
    main()

