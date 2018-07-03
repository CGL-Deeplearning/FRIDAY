import random
import sys


file_name = sys.argv[1]

with open(file_name, "r") as ins:
    for line in ins:
        line_for_print = line.rstrip()
        line = line.rstrip()
        if not line:
            continue
        line = line.split(',')
        gt = line[3]
        if gt[2] == '0':
            if random.random() < 0.1:
                print(line_for_print)
        else:
            print(line_for_print)
