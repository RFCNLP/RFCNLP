for protocol in ["BGPv4", "DCCP", "LTP", "PPTP", "SCTP", "TCP"]:
    num_controls_test = 0; num_controls_train = 0
    data_file = "rfcs-bio/{}_phrases.txt".format(protocol)
    with open(data_file) as fp:
        for line in fp:
            if line.startswith("END-OF-CONTROL"):
                num_controls_test += 1
    data_file = "rfcs-bio/{}_phrases_train.txt".format(protocol)
    with open(data_file) as fp:
        for line in fp:
            if line.startswith("END-OF-CONTROL"):
                num_controls_train += 1

    print(protocol, num_controls_test, num_controls_train)
