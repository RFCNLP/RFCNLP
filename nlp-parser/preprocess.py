from xml.etree import ElementTree as ET
import os
import re
import argparse
import nltk
from nltk_opennlp.chunkers import OpenNLPChunker
from nltk_opennlp.taggers import OpenNLPTagger

def cleanFile(file):
    filename, extension = file.rsplit(".", 1)
    newfile = filename + "-clean." + extension
    with open(file, "r") as fr:
        all_text = fr.read()

        # Remove headers, footnotes and page numbers
        all_text = re.sub(r'Kohler, et al.\s+Standards Track\s+\[Page \w+\]', "", all_text)
        all_text = re.sub(r'RFC 4340\s+Datagram Congestion Control Protocol (DCCP)\s+March 2006', "", all_text)
        all_text = re.sub(r'Stewart\s+Standards Track\s+\[Page \w+\]', "", all_text)
        all_text = re.sub(r'RFC 4960\s+Stream Control Transmission Protocol\s+September 2007', "", all_text)
        all_text = re.sub(r'Ramadas, et al.\s+Experimental\s+\[Page \w+\]', "", all_text)
        all_text = re.sub(r'RFC 5326\s+LTP - Specification\s+September 2008', "", all_text)
        all_text = re.sub(r'Rekhter, et al.\s+Standards Track\s+\[Page \w+\]', "", all_text)
        all_text = re.sub(r'RFC 4271\s+BGP-4\s+January 2006', "", all_text)
        all_text = re.sub(r'Hamzeh, et al.\s+Informational\s+\[Page \w+\]', "", all_text)
        all_text = re.sub(r'RFC 2637\s+Point-to-Point Tunneling Protocol (PPTP)\s+July 1999', "", all_text)
        all_text = re.sub(r'September 1981', "", all_text)
        all_text = re.sub(r'Transmission Control Protocol', "", all_text)
        all_text = re.sub(r'Functional Specification', "", all_text)
        all_text = re.sub(r'\[Page \w+\]', "", all_text)

        # remove newlines if they don't occur after or before another newline
        all_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', all_text)
        all_text = all_text.replace("\f", "").replace("\t", "")
       
        if re.match('.*Rekhter.*', all_text):
            print("here2")
            exit()

        with open(newfile, "w") as fw:
            fw.write(all_text)
        '''
        with open(newfile, "w") as fw:
            for line in fr:
                #print(line)
                l = line.replace("\n", "").replace("\f", "").replace("\t", "")
                #print(l)
                #print("----")
                if l:
                    fw.write(l + "\n")
        '''
    return newfile

'''
    This method was developed by Haozhe in our old code to correct some tokenization issues.
'''
def tokenize(tokenizer, line):
    tokens = tokenizer.tokenize(nltk.word_tokenize(line))
    tokens = [tok.replace('\'', '`') for tok in tokens]
    for i, token in enumerate(tokens):
        if len(token) > 3 and (token.endswith('.') or token.endswith(',')):
            # NLTK sometimes fail to tokenize a word correctly. For example 'transition.'
         tokens[i:i + 1] = token[:-1], token[-1]  # replace  'transition.' with 'transition' and '.'
        # fix some inconsistency in tokenizing
        elif token == '`s':
            tokens[i:i + 1] = '`', 's'
        elif token == 'n`t':
            tokens[i:i + 2] = 'n', '`', 't'
        elif token == 'A.':
            tokens[i:i + 1] = 'A', '.'
        elif token == '`Cncld':
            tokens[i:i + 1] = '`', 'Cncld'
        elif token == '`Max.Init.Retransmits':
            tokens[i:i + 5] = '`', 'Max', '.', 'Init', '.', 'Retransmits'
        elif token == 'it.':
            tokens[i:i + 1] = 'it', '.'
    return tokens

def recursive_parent(elem, fp, tokenizer, args, tt, cp, level_horizontal, level_deep):
    if elem.tag == "control":
        level_horizontal += 1
        control_elems = []
        if elem.text is not None:
            control_elems.append(('O', elem.text, elem.get('relevant'), level_horizontal, level_deep))

        recursive_control(elem, elem.get('relevant'), control_elems, level_horizontal, level_deep + 1)
        write_control(args, control_elems, tokenizer, tt, cp, fp)
    else:
        for child in elem:
            recursive_parent(child, fp, tokenizer, args, tt, cp, level_horizontal, level_deep)

def recursive_tag(elem, parent_tag, attr, accum=[], level_horizontal=0, level_deep=0):
    for child in elem:
        #print(child.tag, child.text)
        if child.text is not None:
            accum.append((parent_tag, child.text, attr, level_horizontal, level_deep))
        recursive_tag(child, parent_tag, attr, accum, level_horizontal, level_deep)
        if child.tail is not None:
            accum.append((parent_tag, child.tail, attr, level_horizontal, level_deep))

def recursive_control(elem, attr, accum=[], level_horizontal=0, level_deep=0):
    for child in elem:
        if child.tag == "control":
            recursive_control(child, child.get('relevant'), accum, level_horizontal, level_deep + 1)
        else:
            if child.text is not None:
                accum.append((child.tag, child.text, attr, level_horizontal, level_deep))
            recursive_tag(child, child.tag, attr, accum, level_horizontal, level_deep)

        if child.tail is not None:
            accum.append(('O', child.tail, attr, level_horizontal, level_deep))

'''
    Generate BIO data in the same style as before
'''
def write_control(args, control_elems, tokenizer, tt, cp, fp):

    prev_tag = None
    for elems in control_elems:
        #print(elems)
        (tag, text, attr, lvl_h, lvl_d) = elems
        #continue
        current_tag = tag
        #print(prev_tag, current_tag)
        #text = re.sub(r'(\n)\s', r'\g<1> &tab;', text)
        #for i in range(0, 100):
        #    text = re.sub(r'(&tab;)\s', r'\g<1>&tab;', text)
        text = re.sub('\n', ' &newline;', text)
        #continue

        tokens = tokenize(tokenizer, text)
        # continue if there are no tokens
        if len(tokens) == 0:
            continue
        #print("INIT TOKENS", tokens)
        pos_tags = [tok[1] for tok in nltk.pos_tag(tokens)]
        #print(tag, "|", text, "|", tokens, "|", pos_tags)

        sentence = " ".join(tokens)

        #print("Sentence | ", str(sentence))

        # Replace characters that break on OpenNLP
        sentence = sentence.replace("_", "")
        sentence = sentence.replace("[", "(")
        sentence = sentence.replace("]", ")")

        init_tag = 'O'; inside_tag = 'O'
        if current_tag != 'O':
            inside_tag = "I-{}".format(current_tag.upper())

        if current_tag != prev_tag and current_tag != 'O':
            init_tag = "B-{}".format(current_tag.upper())
        elif current_tag != 'O':
            init_tag = "I-{}".format(current_tag.upper())


        tree = None
        if not args.no_chunk:
            sentence = tt.tag(sentence)

            if re.match(r'.*\w+.*', str(sentence)):
                try:
                    tree = cp.parse(sentence)
                except:
                    tree = None
            else:
                tree = None


            #print(init_tag)
            if tree:
                tree = str(tree)
                #print(tree)
                idx = 0

                lines = tree.split('\n')
                if len(lines) == 1 and lines[0].startswith('(S ') and lines[0].endswith(')'):
                    lines = re.split(r'(?:\)\s+)|(?:\s+\()', lines[0][3:-1])
                    for i in range(0, len(lines)):
                        if lines[i].startswith('(') and not lines[i].endswith(')'):
                            lines[i] = lines[i] + ")"
                        elif lines[i].endswith('(') and not lines[i].startswith(')'):
                            lines[i] = "(" + lines[i]

                for i, line in enumerate(lines):
                    line = line.strip()
                    #print("LINE", line)

                    if not line.endswith(')') and '/' not in line:
                        # Skip root nodes
                        continue

                    #print("LINE", line)

                    if line.startswith('(') and ' ' in line:
                        first_space = line.index(' ')
                        line = line[first_space+1:-1]
                    #print("LINE", line)
                    chunk_tokens = line.split()
                    chunk_tokens = [t.rsplit('/', 1)[0] for t in chunk_tokens if '/' in t]
                    #if len(chunk_tokens) > 0:
                    #    print("\tCHUNK TOKENS", chunk_tokens)

                    chunk_tag = init_tag
                    if idx > 0:
                        chunk_tag = inside_tag

                    for j, tok in enumerate(chunk_tokens):
                        # correct all cases when &newline; was classified as anything other than a SYMBOL
                        if tokens[idx] == "&newline;":
                            pos_tags[idx] = "SYM"


                        if idx == 0:
                            fp.write(tokens[idx] + ' ======= ' + init_tag + ' ======= ' + pos_tags[idx] + ' ======= ' + str(attr == "true") + ' ======= ' + str(lvl_h) + ' ======= ' + str(lvl_d) + '\n')
                        else:
                            fp.write(tokens[idx] + ' ======= ' + inside_tag + ' ======= ' + pos_tags[idx] + ' ======= ' + str(attr == "true") + ' ======= ' + str(lvl_h) + ' ======= ' + str(lvl_d) + '\n')
                            #init_tag = inside_tag

                        idx += 1

                    fp.write('END-OF-CHUNK' + ' ======= ' + chunk_tag + ' ======= ' + str(lvl_h) + ' ======= ' + str(lvl_d) + '\n')
        if not tree:
            # create a single chunk
            for idx, tok in enumerate(tokens):
                # correct all cases when &newline; was classified as anything other than a SYMBOL
                if tokens[idx] == "&newline;":
                    pos_tags[idx] = "SYM"

                if idx == 0:
                    fp.write(tokens[idx] + ' ======= ' + init_tag + ' ======= ' + pos_tags[idx] + ' ======= ' + str(attr == "true") + ' ======= ' + str(lvl_h) + ' ======= ' + str(lvl_d) +'\n')
                else:
                    fp.write(tokens[idx] + ' ======= ' + inside_tag + ' ======= ' + pos_tags[idx] + ' ======= ' + str(attr == "true") + ' ======= ' + str(lvl_h) + ' ======= ' + str(lvl_d) +'\n')
            if not args.no_chunk:
                fp.write('END-OF-CHUNK' + ' ======= ' + init_tag + ' ======= ' + str(lvl_h) + ' ======= ' + str(lvl_d) +'\n')

        prev_tag = current_tag
    fp.write('END-OF-CONTROL' + ' ======= ' + 'END_CONTROL' + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=str,  help='protocol', required=True)
    parser.add_argument('--no_chunk', default=False, action='store_true')
    args = parser.parse_args()
    
    tt = OpenNLPTagger(language='en',
                   path_to_bin='opennlp_python/apache-opennlp/bin',
                   path_to_model='opennlp_python/opennlp_models/en-pos-maxent.bin')

    cp = OpenNLPChunker(path_to_bin='opennlp_python/apache-opennlp/bin',
                    path_to_chunker='opennlp_python/opennlp_models/en-chunker.bin')

    mwes = [('<', '-'),
            ('<', '<'),
            ('(', '<', ')'),
            ('|', '<', '|'),
            (':', '='),
            ('=', '='),
            ('<', '='),
            ('>', '='),
            ('!', '='),
            ('&', 'newline', ';')]

    tokenizer = nltk.MWETokenizer(mwes=mwes, separator='')

    #PROTOCOLS = ['BGPv4', 'DCCP', 'LTP', 'PPTP', 'SCTP', 'TCP']
    #for protocol in PROTOCOLS:
    print(args.protocol)
    if not args.no_chunk:
        fp = open("rfcs-bio/{}.txt".format(args.protocol), "w")
        fp_train = open("rfcs-bio/{}_train.txt".format(args.protocol), "w")
    else:
        fp = open("rfcs-bio/{}_no_chunk.txt".format(args.protocol), "w")
        fp_train = open("rfcs-bio/{}_no_chunk_train.txt".format(args.protocol), "w")

    rfc = os.path.join("rfcs-annotated-tidied", "{}.xml".format(args.protocol))
    xml = ET.parse(cleanFile(rfc))

    # Over-generating to learn
    for i, control in enumerate(xml.iter('control')):
        control_elems = []
        if control.text is not None:
            control_elems.append(('O', control.text, control.get('relevant'), 0, 0))
        recursive_control(control, control.get('relevant'), control_elems)
        write_control(args, control_elems, tokenizer, tt, cp, fp_train)
    fp_train.close()

    # Without over generation to predict
    level_horizontal = 0; level_deep = 0
    for child in xml.getroot():
        recursive_parent(child, fp, tokenizer, args, tt, cp, level_horizontal, level_deep)
    fp.close()

if __name__ == "__main__":
    main()
