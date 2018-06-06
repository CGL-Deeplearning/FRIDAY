import sys
from collections import OrderedDict
import matplotlib.pyplot as plt

file_name = sys.argv[1]
type_dict = {}
type_dict['SNP'] = 0
type_dict['IN'] = 0
type_dict['DEL'] = 0
with open(file_name, "r") as ins:
    for line in ins:
        line = line.rstrip()
        if not line:
            continue
        line = line.split('\t')
        if len(line) > 1 and line[1] == 'VCF MISMATCH':
            chrm, pos, qual, ref, alt, genotype, rec_type = line[3], line[4], line[5], line[6], line[7], line[8], line[9]
            # print(chrm, pos, qual, ref, alt, genotype, rec_type)
            type_dict[rec_type] += 1
        # if len(line) > 1 and line[1] == 'BAM MISMATCH':
        #     print(line)
print(type_dict)
fig, ax = plt.subplots()

dictionary2 = OrderedDict(sorted(type_dict.items(), key=lambda t: t[0]))

ax.bar(range(len(dictionary2)), dictionary2.values(), align='center')
plt.xticks(range(len(dictionary2)), dictionary2.keys())

for i, v in enumerate(dictionary2.values()):
    ax.text(i, v + 500, str(v) + ": " + str(round(v*100/sum(dictionary2.values()), 2)) + "%",
            fontweight='bold', ha='center', fontsize=8)
plt.savefig(file_name.split('/')[-1].split('.')[0]+"_Visualized.png", dpi=400)