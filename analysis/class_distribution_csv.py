import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from collections import OrderedDict


sns.set(color_codes=True)

file_name = sys.argv[1]

dictionary = {}
dictionary['0'] = 0
dictionary['1'] = 0
dictionary['2'] = 0
dictionary['3'] = 0
dictionary['4'] = 0
dictionary['5'] = 0
# dictionary['3'] = 0
all_zero_sequences = 0
total = 0
with open(file_name, "r") as ins:
    for line in ins:
        line = line.rstrip()
        if not line:
            continue
        line = line.split(',')
        gts = line[3]
        all_zero = True
        for gt in gts:
            dictionary[gt] += 1
            if gt != '0':
                all_zero = False
                break
        if all_zero:
            all_zero_sequences += 1
        total += 1
print('All zeros: ', all_zero_sequences)
print('Total: ', total, " Percent: ", int(all_zero_sequences/total))
print('0/0', dictionary['0'])
print('0/1', dictionary['1'])
print('1/1', dictionary['2'])
print('0/2', dictionary['3'])
print('2/2', dictionary['4'])
print('1/2', dictionary['5'])

fig, ax = plt.subplots()

dictionary2 = OrderedDict(sorted(dictionary.items(), key=lambda t: t[0]))

ax.bar(range(len(dictionary2)), dictionary2.values(), align='center')
plt.xticks(range(len(dictionary2)), ['0/0', '0/1', '1/1', '0/2', '2/2', '1/2'])

for i, v in enumerate(dictionary2.values()):
    ax.text(i, v + 500, str(v) + ": " + str(round(v*100/sum(dictionary2.values()), 2)) + "%",
            fontweight='bold', ha='center', fontsize=8)
plt.savefig(file_name.split('/')[-1].split('.')[0]+"_Visualized.png", dpi=400)