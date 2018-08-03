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
        gts = line[3]
        all_zero = True
        for gt in gts:
            if gt != '0':
                all_zero = False
                break
        if all_zero:
            if random.random() < 0.1:
                print(line_for_print)
        else:
            print(line_for_print)
