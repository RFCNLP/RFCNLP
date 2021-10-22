#!/usr/bin/env python3

import os
import re
import numpy as np

def remove_headers(text):
    text = re.sub(r'Kohler, et al\.              Standards Track                   \[Page \d+]', "", text)
    text = re.sub(r'Ramadas, et al\.               Experimental                     \[Page \d+]', "", text)
    text = re.sub(r'RFC 4340      Datagram Congestion Control Protocol \(DCCP\)     March 2006', "", text)
    text = re.sub(r'RFC 5326                  LTP - Specification             September 2008', "", text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text

'''
This method is intended to work over tags of a single type only
'''
def get_spans(text, tag_type):
    closing_tag = "</{0}>".format(tag_type)
    iterator = re.finditer('</*{0}( \w+=\w+)*>'.format(tag_type), text)
    tags_all = [match.span()[0] for match in iterator]
    opening = []; closing = []; empty_positions = []

    for span in tags_all:
        is_open_this = re.match("<\w", text[span:span+2]) is not None

        if is_open_this:
            opening.append(span)
            closing.append(None)
            empty_positions.append(len(closing)-1)
        else:
            '''
            if len(empty_positions) == 0:
                for (a,b) in zip(opening, closing):
                    print(text[a:b+len(closing_tag)])
                    print("====")
                print(text[span-100:span])
                exit()
            print("opening", opening)
            print("closing", closing)
            print("empty_positions", empty_positions)
            '''
            if len(empty_positions) > 0:
                last_empty = empty_positions[-1]
                closing[last_empty] = span
                empty_positions = empty_positions[:-1]

    spans = []
    for (a, b) in zip(opening, closing):
        try:
            #print(text[a:b+len(closing_tag)])
            #print("====")
            spans.append(text[a:b+len(closing_tag)])
        except:
            print(text[a:a+100])
            print("start={}; end={}".format(a,b))
            exit()
    return spans

def count_tags(text, tag_type):
    tag = "<{0}( \w+=\"\w+\")*>".format(tag_type)
    tag_open = len(re.findall(tag, text))

    tag = "</{0}>".format(tag_type)
    tag_close = len(re.findall(tag, text))
    return tag_open, tag_close


def process_file(path):
    STATE_TAGS = ["action", "variable", "timer", "error", "transition", "trigger",
                  "def_event", "ref_event", "def_state", "ref_state"]

    with open(path) as fp:
        text = fp.read()
        text = remove_headers(text)

        print("=== CONTROL TAGS:")
        tag_open, tag_close = count_tags(text, "control")
        print("\t{0} | open: {1}, close: {2}".format("control", tag_open, tag_close))
        #return


        print("=== STATE MACHINE TAGS:")
        state_spans = {}
        for tag_type in STATE_TAGS:
            tag_open, tag_close = count_tags(text, tag_type)
            print("\t{0} | open: {1}, close: {2}".format(tag_type, tag_open, tag_close))
            state_spans[tag_type] = get_spans(text, tag_type)
            #print(state_spans[tag_type])

        # Position of all control tags for parsing spans
        control_spans = get_spans(text, "control")

        # Check the number of tags in deepest controls only
        # Keep in mind that if we remove the if then some counts will be duplicated in nested cases
        tags_in_spans = {}

        deepest_control_spans = [span_text for span_text in control_spans if span_text.count('<control') == 1 ]

        sm_per_span = []; words_per_span = []; sentences_per_span = []
        for span_text in deepest_control_spans:

            # Skip when not deepest
            if span_text.count('<control') > 1:
                continue

            sm_per_span.append(len(re.findall('<(a|e|t|v|r|d)', span_text)))
            clean_text = re.sub('<\w+>', "", span_text)
            clean_text = re.sub('</\w+>', "", clean_text)
            words = clean_text.split()
            sentences = clean_text.split('.')

            #if len(words) >= 100:
            #    print(words)
            #    print("====")
            words_per_span.append(len(words))
            sentences_per_span.append(len(sentences))


            #print(span_text)
            #print( "====")

            for tag_type in STATE_TAGS:
                if tag_type not in tags_in_spans:
                    tags_in_spans[tag_type] = []

                tags_spans = get_spans(span_text, tag_type)
                tags_in_spans[tag_type] += tags_spans

        print("=== DEEPEST CONTROL SPANS", len(deepest_control_spans))
        print("=== SM PER CONTROL (DEEPEST)")
        # print(sm_per_span)
        # print("\tMax", max(sm_per_span))
        # print("\tMin", min(sm_per_span))
        # print("\tMean", np.mean(sm_per_span))
        # print("=== WORDS PER CONTROL (DEEPEST)")
        # print("\tMax", max(words_per_span))
        # print("\tMin", min(words_per_span))
        # print("\tMean", np.mean(words_per_span))
        # print("=== SENTENCES PER CONTROL (DEEPEST)")
        # print("\tMax", max(sentences_per_span))
        # print("\tMin", min(sentences_per_span))
        # print("\tMean", np.mean(sentences_per_span))

        print("=== STATE MACHINE TAGS INSIDE DEEPEST CONTROL SPANS")
        for tag_type in tags_in_spans:
            print("\t{0} | {1}".format(tag_type, len(tags_in_spans[tag_type])))


def main():
    FILES = ["BGPv4.txt", "PPTP.txt", "TCP.txt", "SCTP.txt", "DCCP.txt", "LTP.txt"]
    for protocol in FILES:
        path = os.path.join("rfcs-annotated", protocol)
        print(path)
        process_file(path)


if __name__ == "__main__":
    main()
