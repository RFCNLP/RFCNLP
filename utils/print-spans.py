#!/usr/bin/env python3

import sys
import os
import re
from xml.etree import ElementTree as ET

state_machine_tags = ["transition","variable","timer","error","action"]

def cleanFile(file):
    filename, extension = file.rsplit(".", 1)
    newfile = filename + "-clean." + extension
    with open(file, "r") as fr:
        with open(newfile, "w") as fw:
            for line in fr:
                l = line.replace("\n", "").strip().replace("\f", "")
                if l:
                    fw.write(l + "\n")
    return newfile

def print_state_machine_spans(root):
    for child in root:
        if child.tag in state_machine_tags:
            print(child.tag.upper() + ":: " + child.text)
        else:
            for grandchild in child:
                print_state_machine_spans(grandchild)


if __name__ == "__main__":
    rfc = sys.argv[1]
    parser = ET.XMLParser(encoding="utf-8")
    xml = ET.parse(cleanFile(rfc), parser=parser).getroot()
    xml = ET.parse(rfc, parser=parser).getroot()
    print_state_machine_spans(xml)
