import uuid
import subprocess
from pathlib import Path
import os.path

def analyzeDCCPattack(attackFileName):

    for harnessFile in glob("src/dccp.harness.files/harness*.pml"):

        # rm tmp.* pan *.tmp pan.pre
        _ = subprocess.Popen(
                    ['rm', 'tmp.*', 'pan', '*.tmp', 'pan.pre'], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    encoding='utf8')

        newfile = None

        newFileName = "strategies.DCCP."

        if "linear" in attackFileName:
            newFileName += "linear."
        elif "bert" in attackFileName:
            newFileName += "bert."
        elif "korg-anon" in attackFileName:
            newFileName += "canonical."
        else:
            newFileName += "gold."

        if "phi1" in attackFileName or "DCCP.1_True" in attackFileName:
            newFileName += "phi1."
        elif "phi2" in attackFileName or "DCCP.2_True" in attackFileName:
            newFileName += "phi2."
        elif "phi3" in attackFileName or "DCCP.3_True" in attackFileName:
            newFileName += "phi3."
        elif "phi4" in attackFileName or "DCCP.4_True" in attackFileName:
            newFileName += "phi4."

        newFileName += attackFileName.split("/")[-1].split("_WITH")[0] 

        newFileName += ".phi" +  harnessFile.split(".pml").split("harness")[2]

        newFileName += ".strategy"

        newFileName = newFileName.replace("/", ".")

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
                2 : 'Request',      #define RequestState   2
                3 : 'Respond',      #define RespondState   3
                4 : 'Part-Open',    #define PartOpenState  4
                5 : 'Open',         #define OpenState      5
                6 : 'Close-Req',    #define CloseReqState  6
                7 : 'Closing',      #define ClosingState   7
                8 : 'Time-Wait',    #define TimeWaitState  8
                -1 : '(terminated)' #define EndState       -1
            }

            with open(newFileName, 'w') as fw:

                fw.write("Strategy for: " + attackFileName + "\n")
                
                for line in stdout2.split("\n"):

                    lline = line.strip()
                    wrtline = None
                    
                    if ('state[0] = ' in lline or 'state[1] = ' in lline) and \
                       (not "[state" in line)                             and \
                       (not "before_state" in line):
                        
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