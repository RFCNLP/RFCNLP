from xml.etree import ElementTree as ET
import os
import re
import argparse
import nltk
from collections import Counter
from nltk_opennlp.chunkers import OpenNLPChunker
from nltk_opennlp.taggers import OpenNLPTagger

def cleanFile(file):
    filename, extension = file.rsplit(".", 1)
    newfile = filename + "-clean." + extension
    with open(file, "r") as fr:
        all_text = fr.read()

        # Remove headers, footnotes and page numbers
        all_text = re.sub(r'Kohler, et al.\s+Standards Track\s+\[Page \w+\]', "", all_text)
        all_text = re.sub(r'RFC\s+4340\s+Datagram\s+Congestion\s+Control\s+Protocol\s+(DCCP)\s+March\s+2006', "", all_text)
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
        all_text = re.sub(r'\[RFC\d+\]', "", all_text)
        # Remove comments
        #all_text = re.sub(r'\/\*[\s\S]*?\*\/', "", all_text)
        
        # remove newlines if they don't occur after or before another newline, or after or before some identation
        lines = all_text.split('\n')
        ret_text = lines[0].strip(); indenting = False
        for i in range(1, len(lines)):
            prev_line = lines[i-1]
            curr_line = lines[i]

            leading_spaces_prev = len(prev_line) - len(prev_line.lstrip())
            leading_spaces_curr = len(curr_line) - len(curr_line.lstrip())

            # Check whether we are in pseudo-code territory and turn on a indenting flag
            if leading_spaces_curr > leading_spaces_prev and re.match(r'(<control relevant="true">)*(<trigger>)*(if|otherwise).*', prev_line.lstrip().lower()):
                indenting = True
            elif leading_spaces_curr > leading_spaces_prev:
                indenting = False

            '''
            if curr_line.lstrip().lower().startswith("<action>calculate new value</action>"):
                print(indenting)
            if curr_line.lstrip().lower().startswith("<action>send confirm l on a future packet</action>"):
                print(indenting)
                exit()
            '''
            
            if ((leading_spaces_curr > leading_spaces_prev or leading_spaces_curr < leading_spaces_prev) \
                and not (prev_line.lstrip().startswith('o  '))) or indenting:
                ret_text += "\n"
            
            ret_text += " " + curr_line.strip()

        #all_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', all_text)
        all_text = ret_text
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

def recursive_tag(elem, parent_tag, attr, accum=[], level_horizontal=0, level_deep=0):
    for child in elem:
        #print(child.tag, child.text)
        if child.text is not None:
            accum.append((parent_tag, child.text, attr, level_horizontal, level_deep))
        recursive_tag(child, parent_tag, attr, accum, level_horizontal, level_deep)
        if child.tail is not None:
            accum.append((parent_tag, child.tail, attr, level_horizontal, level_deep))


def recursive_parent(elem, fp, tokenizer, args, level_horizontal, level_deep):
    if elem.tag == "control":
        level_horizontal += 1
        control_elems = []
        if elem.text is not None:
            control_elems.append(('O', elem.text, elem.get('relevant'), level_horizontal, level_deep))

        recursive_control(elem, elem.get('relevant'), control_elems, level_horizontal, level_deep + 1)
        write_control(args, control_elems, tokenizer, fp)
        #level_horizontal += 1
    else:
        for child in elem:
            recursive_parent(child, fp, tokenizer, args, level_horizontal, level_deep)

def recursive_control(elem, attr, accum=[], level_horizontal=0, level_deep=0):
    for child in elem:
        if child.tag == "control":
            level_horizontal += 1
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
def write_control(args, control_elems, tokenizer, fp):

    prev_tag = None
    all_tags = []; all_text = []; all_attrs = []; all_pos = []; all_levels_h = []; all_levels_d = []
    for (tag, text, attr, level_h, level_d) in control_elems:
        text = re.sub('\n', ' &newline;', text)
        tokens = tokenize(tokenizer, text)
        # continue if there are no tokens
        if len(tokens) == 0:
            continue
        pos_tags = [tok[1] for tok in nltk.pos_tag(tokens)]
        fsm_tags = [tag] * len(tokens)
        fsm_levels_h = [level_h] * len(tokens)
        fsm_levels_d = [level_d] * len(tokens)
        attrs = [attr] * len(tokens)
        all_text += tokens
        all_tags += fsm_tags
        all_attrs += attrs
        all_pos += pos_tags
        all_levels_h += fsm_levels_h
        all_levels_d += fsm_levels_d

    splits = ['.', ',', ';', '&newline;', '/*', '*/']
    open_par = False;
    chunk_tags = []; prev_tag = None
    chunk_words = []

    for i in range(0, len(all_text)):
        tok = all_text[i]; tag = all_tags[i]; attr = all_attrs[i]; pos = all_pos[i]; lvl_h = all_levels_h[i]; lvl_d = all_levels_d[i]
        if tok in '(':
            open_par = True
        if tok == ')':
            open_par = False

        fp.write(tok + ' ======= ' + tag + ' ======= ' + pos + ' ======= ' + str(attr == "true") + ' ======= ' + str(lvl_h) + ' ======= ' + str(lvl_d) + '\n')
        chunk_tags.append(tag)
        chunk_words.append(tok)
        if ((tok in splits or i == len(all_text) - 1) and not open_par) or\
            (args.split_on_conditional and i < len(all_text) - 1 and all_text[i+1].lower() in ["if", "otherwise", "then", "when", "while"] and i != 0 and not open_par):
            #print(chunk_words)
            c = Counter(chunk_tags)
            value, count = c.most_common()[0]
            if value == 'O' and len(c.most_common()) > 1:
                value, count = c.most_common()[1]

            chunk_tag = value
            if chunk_tag != 'O':
                chunk_tag = "B-" + chunk_tag.upper()

            fp.write('END-OF-CHUNK' + ' ======= ' + chunk_tag + ' ======= ' + str(lvl_h) + ' ======= ' + str(lvl_d) + '\n')
            chunk_tags = []; chunk_words = []

    #print(" ".join(all_text))
    #print(len(all_text), len(all_tags), len(all_attrs), len(all_pos))
    #print("============")
    #exit()
    fp.write('END-OF-CONTROL' + ' ======= ' + 'END_CONTROL' + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=str,  help='protocol', required=True)
    parser.add_argument('--split_on_conditional', default=False, action='store_true')
    args = parser.parse_args()

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
    fp = open("rfcs-bio/{}_phrases.txt".format(args.protocol), "w")
    fp_train = open("rfcs-bio/{}_phrases_train.txt".format(args.protocol), "w")
    rfc = os.path.join("rfcs-annotated-tidied", "{}.xml".format(args.protocol))
    xml = ET.parse(cleanFile(rfc))

    # Over-generating to learn
    for i, control in enumerate(xml.iter('control')):
        control_elems = []
        if control.text is not None:
            control_elems.append(('O', control.text, control.get('relevant'), 0, 0))
        recursive_control(control, control.get('relevant'), control_elems)
        write_control(args, control_elems, tokenizer, fp_train)
    fp_train.close()

    # Without over generation to predict
    level_horizontal = 0; level_deep = 0
    for child in xml.getroot():
        recursive_parent(child, fp, tokenizer, args, level_horizontal, level_deep)
    fp.close()

if __name__ == "__main__":
    main()
