import argparse
import os
'''
   Track indentation and sentence boundaries of control statements
   to get their scope.
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=str,  help='protocol', required=True)
    args = parser.parse_args()

    # Find which files we have annotations for
    annotated_protocols = [filename[:-4] for filename in os.listdir("rfcs-annotated-tidied") if filename.endswith(".xml")]
    if args.protocol not in annotated_protocols:
        print("Protocol is not supported")
        exit(-1)

    print(args.protocol)
    fp_ann = open('rfcs-annotated-tidied/{}.xml'.format(args.protocol))
    indent_range = set()

    for line in fp_ann:
        indent = len(line) - len(line.lstrip())
        if "<control" in line:
            indent_range.add(indent)

    indent_range = list(indent_range)
    indent_range.sort()
    projected_range = list(range(1, len(indent_range)+1))
    print(indent_range, projected_range)

    fp_ann = open('rfcs-annotated-tidied/{}.xml'.format(args.protocol))
    fp_out = open('rfcs-annotated-tidied/{}-format.xml'.format(args.protocol), "w")
    sent_count = 0
    for line in fp_ann:
        indent = len(line) - len(line.lstrip())
        if "<control" in line:
            index_of_indent = indent_range.index(indent)
            indent_proj = projected_range[index_of_indent]

            sent_count += 1
            line = line.replace('<control ', '<control indent="{}" count="{}" '.format(indent_proj, sent_count))
        fp_out.write(line)


if __name__ == "__main__":
    main()
