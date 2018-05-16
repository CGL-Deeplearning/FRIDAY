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
# dictionary['3'] = 0
with open(file_name, "r") as ins:
    for line in ins:
        line = line.rstrip()
        if not line:
            continue
        line = line.split(',')
        gt = line[2]
        dictionary[gt] += 1


fig, ax = plt.subplots()

dictionary2 = OrderedDict(sorted(dictionary.items(), key=lambda t: t[0]))

ax.bar(range(len(dictionary2)), dictionary2.values(), align='center')
plt.xticks(range(len(dictionary2)), ['Hom', 'Het', 'Hom-alt'])

for i, v in enumerate(dictionary2.values()):
    ax.text(i, v + 500, str(v) + ": " + str(round(v*100/sum(dictionary2.values()), 2)) + "%",
            fontweight='bold', ha='center', fontsize=8)
plt.savefig(file_name.split('/')[-1].split('.')[0]+"_Visualized.png", dpi=400)