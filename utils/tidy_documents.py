#!/usr/bin/env python3

import os
import re

tags = ['def_state', 'def_var', 'control', 'trigger', 'action', 'error', 'timer',
        'transition', 'variable', 'state', 'event-upper', 'event-lower',
        'event-internal', 'arg', 'def_event', 'ref_event', 'ref_state', 'arg_source', 'arg_target', 'arg_source_target', 'arg_intermediate']

tags_with_attributes = ['control relevant=\"true\"', 'control relevant=\"false\"', 'control relevant=\"unsure\"',
                        'action type=send', 'action type=receive', 'action type=issue',

                        # Need to enumerate these options, because variable lengths
                        # don't work with regex backtracking. Since id is a monotonically
                        # increasing integer identifier, it's unlikely any RFCs will
                        # need more than three digits.
                        'def_event id=\"\w\"', 'def_event id=\"\w\w\"', 'def_event id=\"\w\w\w\"',
                        'ref_event type="send" id=\"\w\"', 'ref_event type="send" id=\"\w\w\"',
                        'ref_event type="send" id=\"\w\w\w\"',
                        'ref_event type="receive" id=\"\w\"', 'ref_event type="receive" id=\"\w\w\"',
                        'ref_event type="receive" id=\"\w\w\w\"',
                        'ref_event type="compute" id=\"\w\"', 'ref_event type="compute" id=\"\w\w\"',
                        'ref_event type="compute" id=\"\w\w\w\"',

                        'def_state id=\"\w\"', 'def_state id=\"\w\w\"', 'def_state id=\"\w\w\w\"',
                        'ref_state id=\"\w\"', 'ref_state id=\"\w\w\"', 'ref_state id=\"\w\w\w\"']


for protocol in ["BGPv4", "DCCP", "LTP", "PPTP", "SCTP", "TCP"]:
    input_file = "rfcs-annotated/{}.txt".format(protocol)
    output_file = "rfcs-annotated-tidied/{}.xml".format(protocol)

    with open(input_file) as fp:
        document = fp.read()

        # Find all ampersands
        document = re.sub(r'&', '&amp;', document)

        # Findall "<" that are not part of our tags
        regex_fields = "|".join(['/*' + t for t in tags])
        regex_string = f'<(?!({regex_fields}))'
        regex = re.compile(regex_string)
        document = re.sub(regex, '&lt;', document)

        # Findall ">" that are not part of our tags
        regex_string = "".join(['(?<!' + t + ')' for t in tags]) + \
                       "".join(['(?<!' + t + ')' for t in tags_with_attributes]) + '>'
        regex = re.compile(regex_string)
        document = re.sub(regex, '&gt;', document)

        # Replace arguments with quotes
        document = re.sub(r'<control relevant=true>', '<control relevant="true">', document)
        document = re.sub(r'<control relevant=false>', '<control relevant="false">', document)
        document = re.sub(r'<control relevant=unsure>', '<control relevant="unsure">', document)
        document = re.sub(r'<action type=send>', '<action type="send">', document)
        document = re.sub(r'<action type=receive>', '<action type="receive">', document)
        document = re.sub(r'<action type=issue>', '<action type="issue">', document)



        document = "<p>" + document + "</p>"

        with open(output_file, "w") as fp:
            fp.write(document)
