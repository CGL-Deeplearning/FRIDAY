import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from collections import OrderedDict, defaultdict


sns.set(color_codes=True)

file_name = sys.argv[1]

dictionary = defaultdict(int)
window_size = 0
total = 0
with open(file_name, "r") as ins:
    for line in ins:
        line = line.rstrip()
        if not line:
            continue
        line = line.split(',')
        gts = line[3]
        window_size = len(gts)
        for i, gt in enumerate(gts):
            if gt != '0':
                dictionary[i] += 1
                total += 1

x_ticks = []
for i in range(0, window_size):
    x_ticks.append(str(i))
    print("Total non-hom in position", i, ":\t", dictionary[i], "(" + str(round((dictionary[i] * 100)/total, 2)) + "%)")

fig, ax = plt.subplots()

dictionary2 = OrderedDict(sorted(dictionary.items(), key=lambda t: t[0]))
ax.bar(range(len(dictionary2)), dictionary2.values(), align='center')
plt.xticks(range(len(dictionary2)), x_ticks)

for i, v in enumerate(dictionary2.values()):
    ax.text(i, v + 500, str(round(v*100/sum(dictionary2.values()), 2)) + "%",
            fontweight='bold', ha='center', fontsize=8)
plt.savefig(file_name.split('/')[-1].split('.')[0]+"_Visualized.png", dpi=400)