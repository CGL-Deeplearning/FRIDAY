import random
import sys


file_name = sys.argv[1]

with open(file_name, "r") as ins:
    for line in ins:
        line = line.rstrip()
        if not line:
            continue
        line = line.split(',')
        gt = line[3]
        if gt == '0':
            if random.random() < 0.05:
                print(line)
        else:
            print(line)
