from xml.etree import ElementTree as ET
import os
import re

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

def main():
    fp_states = open("rfcs-definitions/def_states.txt", "w")
    fp_vars = open("rfcs-definitions/def_vars.txt", "w")
    fp_events = open("rfcs-definitions/def_events.txt", "w")

    for protocol in ["BGPv4", "DCCP", "LTP", "PPTP", "SCTP", "TCP"]:
        rfc = os.path.join("rfcs-annotated-tidied", "{}.xml".format(protocol))
        xml = ET.parse(cleanFile(rfc))

        for i, state_def in enumerate(xml.iter('def_state')):
            fp_states.write("{}\t{}\t{}\n".format(protocol, state_def.text, state_def.get("id")))

        for i, var_def in enumerate(xml.iter('def_var')):
            fp_vars.write("{}\t{}\n".format(protocol, var_def.text))

        for i, event_def in enumerate(xml.iter('def_event')):
            fp_events.write("{}\t{}\t{}\n".format(protocol, event_def.text, event_def.get("id")))

    fp_states.close()
    fp_vars.close()
    fp_events.close()

if __name__ == "__main__":
    main()
