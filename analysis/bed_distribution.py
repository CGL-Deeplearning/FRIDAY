import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import seaborn as sns

sns.set(color_codes=True)

file_name = sys.argv[1]

interval_sizes = []
total_bases = 0
total_we_can_use = 0
# dictionary['3'] = 0
with open(file_name, "r") as ins:
    for line in ins:
        line = line.rstrip()
        if not line:
            continue
        line = line.split('\t')
        interval_start, interval_end = int(line[1]), int(line[2])
        interval_len = interval_end - interval_start
        if interval_len < 300:
            interval_sizes.append((interval_end - interval_start))
        else:
            interval_sizes.append(300)
        total_bases += (interval_end - interval_start)
        if (interval_end - interval_start) >= 300:
            total_we_can_use += (interval_end - interval_start)

total_miss = sum(i < 300 for i in interval_sizes)
print("Total bases we can use: ", total_we_can_use, "/", total_bases, str(total_we_can_use*100/total_bases) + "%")
print("Total bases: ", total_bases)
print("Total usable: ", total_we_can_use)
plt.hist(interval_sizes, normed=True, bins=100)

plt.ylabel('Count')

plt.savefig(file_name.split('/')[-1].split('.')[0]+"_Visualized.png", dpi=400)