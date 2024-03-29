#! /usr/bin/env python
import os
import struct

"""
THIS FILE SORTS THE 5 DOMAINS BELOW INTO THEIR OWN TEXT FILE
"""


"""
Fri 11 Aug 03_00_01 UTC 2017.log
Mon  7 Aug 03_00_01 UTC 2017.log
Mon 14 Aug 03_00_01 UTC 2017.log
Sat 12 Aug 03_00_01 UTC 2017.log
Sun 13 Aug 03_00_01 UTC 2017.log
Thu 10 Aug 03_00_01 UTC 2017.log
Tue  8 Aug 03_00_01 UTC 2017.log
Tue 15 Aug 03_00_01 UTC 2017.log
Wed  9 Aug 03_00_01 UTC 2017.log
Wed 16 Aug 03_00_01 UTC 2017.log
"""


file3 = open("crypto.txt", "a")
file4 = open("tovar.txt", "a")
file5 = open("Dyre.txt", "a")
file6 = open("Nymaim.txt", "a")
file7 = open("Locky.txt", "a")
line = ""
files = os.listdir()
dir_path = os.path.dirname(os.path.realpath(__file__))
for filename in files:
    if filename[-4:] == ".log":
        file2 = open(os.path.join(dir_path, filename), "r")
        #lines = file2.readlines()
        for line in iter(file2):
            domain = line.split(',')[0]
            if "Cryptolocker" in line:
                file3.write(domain + ", Cryptolocker\n")
                continue
            if "Tovar" in line:
                file4.write(domain + ", Tovar\n")
                continue
            if "Dyre" in line:
                file5.write(domain + ", dyre\n")
                continue
            if "Nymaim" in line:
                file6.write(domain + ", nymaim\n")
                continue
            if "Locky" in line:
                file7.write(domain + ", locky\n")
                continue
        file2.close()
file3.close()
file4.close()
file5.close()
file6.close()
file7.close()
