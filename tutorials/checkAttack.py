# checkAttack.py - a useful script to check if a TCP or DCCP
# attack is confirmed or not.
import sys
from src.testDCCP import *
from src.testTCP  import *

if "TCP" in sys.argv[1].upper():
	print("Testing against TCP.")
	analyzeTCPattack(sys.argv[1])
elif "DCCP" in sys.argv[1].upper():
	print("Testing against DCCP.")
	analyzeDCCPattack(sys.argv[1])
else:
	print("Please rename the attack with TCP or " + 
		  "DCCP in the name, depending on which " + 
		  "protocol it was synthesized for, and " + 
		  "try again.")