import argparse
from xml.etree import ElementTree as ET
import os
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol_xml', type=str, required=True)
    parser.add_argument('--output_xml', type=str, required=True)
    args = parser.parse_args()

    xml = ET.parse(args.protocol_xml)

    for i, control in enumerate(xml.iter("control")):
        control_elems = []

if __name__ == "__main__":
    main()
