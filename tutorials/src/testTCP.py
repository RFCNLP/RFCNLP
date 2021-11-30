import uuid
import subprocess
from pathlib import Path
import os.path

def analyzeTCPattack(attackFileName):

    newfile = None

    if "phi6" in attackFileName:
        print("Not analyzing " + attackFileName + " because it's about phi6.")

    harnessFile = "tcp.harness.files/tcp.harness.pml" if len(sys.argv) == 2 else sys.argv[2]

    print("HARNESS = ", harnessFile)

    newFileName = "strategies/TCP/"

    if "linear" in attackFileName:
        newFileName += "linear/"
    elif "bert" in attackFileName:
        newFileName += "bert/"
    elif "korg-anon" in attackFileName:
        newFileName += "canonical/"
    else:
        newFileName += "gold/"

    if "phi1" in attackFileName or "TCP.1_True" in attackFileName:
        newFileName += "phi1/"
    elif "phi2" in attackFileName or "TCP.2_True" in attackFileName:
        newFileName += "phi2/"
    elif "phi3" in attackFileName or "TCP.3_True" in attackFileName:
        newFileName += "phi3/"
    elif "phi4" in attackFileName or "TCP.4_True" in attackFileName:
        newFileName += "phi4/"

    newFileName += attackFileName.split("/")[-1].split("_WITH")[0]

    if len(sys.argv) != 2:
        newFileName += ".phi" + harnessFile.split(".pml")[0].split("/")[1].split("harness")[1]

    newFileName = newFileName.replace("/", ".")

    newFileName += ".strategy"

    if os.path.isfile(newFileName):
        print(newFileName + " is already computed.  Moving swiftly onward!")

    else: 

        with open(attackFileName, "r") as fr:
            with open(harnessFile, "r") as fr2:
                newfile = "tmp." + str(uuid.uuid4()) + ".pml"
                with open(newfile, "w") as fw:
                    fw.write(fr2.read())
                    fw.write("\n")
                    fw.write(fr.read().replace("// N begins here ...",
                                               "// N begins here ...\n\tb = 1;\n"))
                    fw.write("\n")

        result = subprocess.Popen(
                ['spin', '-run', '-a', '-DNOREDUCE', newfile], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                encoding='utf8')

        stdout, stderr = result.communicate()

        print(stdout)
        print("-----------------------------------")

        # There should be a trail file.

        result2 = subprocess.Popen(
            [
                'spin', 
                '-t0', 
                '-s', 
                '-r', 
                '-p', 
                '-g', 
                newfile
            ], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                encoding='utf8')

        stdout2, stderr2 = result2.communicate()

        print(stdout2)

        if "cannot find" in stdout2:

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("ERROR: failed to generate any trail using: " + attackFileName)
            print("\n\n")
            with open(newfile, "r") as fr:
                print(fr.read())
            print("\n\n")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        print("-----------------------------------")

        stateMap = {
            0 : 'Closed',       #define ClosedState    0
            1 : 'Listen',       #define ListenState    1
            2 : 'Syn-Sent',     #define SynSentState   2
            3 : 'Syn-Received', #define SynRecState    3
            4 : 'Established',  #define EstState       4
            5 : 'Fin-Wait-1',   #define FinW1State     5
            6 : 'Close-Wait',   #define CloseWaitState 6
            7 : 'Fin-Wait-2',   #define FinW2State     7
            8 : 'Closing',      #define ClosingState   8
            9 : 'Last-Ack',     #define LastAckState   9
            10 : 'Time-Wait',   #define TimeWaitState  10
            -1 : '(terminated)' #define EndState       -1
        }

        with open(newFileName, 'w') as fw:

            fw.write("Strategy for: " + attackFileName + "\n")
            
            for line in stdout2.split("\n"):

                lline = line.strip()
                wrtline = None
                
                if ('state[0] = ' in lline or 'state[1] = ' in lline) and (not "[state" in line):
                    
                    who = 'peerA' if 'state[0] = ' in lline else 'peerB'
                    where = None
                    try:
                        where = stateMap[int(lline.split('=')[1].strip())]
                        wrtline = who + ' in ' + where
                    
                    except Exception as e:
                        
                        print("Exception when trying to extract int from " + lline)
                        print("Split on = to get: " + lline.split('=')[1])
                        print("Then stripped to get: " + lline.split('=')[1].strip())
                        print("But parse to int failed ...")

                elif 'CYCLE' in lline:

                    wrtline = lline

                elif 'snd!' in lline:

                    whatSent = lline.split('!')[1].strip().replace("]", "")

                    if 'proc  3' in lline:

                        wrtline = 'peerB sends ' + whatSent

                    elif 'proc  2' in lline:

                        wrtline = 'peerA sends ' + whatSent

                elif 'rcv?' in lline:

                    whatReceived = lline.split('?')[1].strip().replace("]", "")

                    if 'proc  3' in lline:

                        wrtline = 'peerB receives ' + whatReceived

                    elif 'proc  2' in lline:

                        wrtline = 'peerA receives ' + whatReceived

                if wrtline != None:

                    fw.write(str(wrtline) + "\n")
                    print(str(wrtline))

            fw.close()

        print("Saved strategy to " + newFileName)