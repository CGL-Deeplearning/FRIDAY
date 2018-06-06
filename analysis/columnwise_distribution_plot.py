import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from collections import OrderedDict
from PIL import Image
import numpy as np
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm
sns.set(color_codes=True)


def get_base_by_color(base):
    """
    Get color based on a base.
    - Uses different band of the same channel.
    :param base:
    :return:
    """
    if 250.0 <= base <= 255.0:
        return 'A'
    if 100.0 <= base <= 105.0:
        return 'C'
    if 180.0 <= base <= 185.0:
        return 'G'
    if 25.0 <= base <= 35.0:
        return 'T'
    if 3.0 <= base <= 7.0:
        return '*'
    if 0.0 <= base <= 3.0:
        return ' '


def analyze_it(arg_list):
    freq_dictionary = defaultdict(lambda: defaultdict(int))
    for arg_tuple in arg_list:
        img, shape, vcf_alt1, vcf_alt2 = arg_tuple
        file = img
        img_file = Image.open(file)
        img_h, img_w, img_c = shape
        np_array_of_img = np.array(img_file.getdata())
        img_file.close()
        img = np.reshape(np_array_of_img, shape)
        img = np.transpose(img, (0, 1, 2))

        ref_string = ''
        for j in range(img_w):
            if img[0][j][0] != 0:
                ref_string += get_base_by_color(img[0][j][0])
            else:
                ref_string += ' '

        for i in range(img_w):
            alt1 = vcf_alt1[i] if vcf_alt1[i] != '-' else ref_string[i]
            alt2 = vcf_alt2[i] if vcf_alt2[i] != '-' else ref_string[i]
            if alt2 < alt1:
                alt1, alt2 = alt2, alt1

            for j in range(1, img_h):
                if img[j][i][0] != 0:
                    base_in_pos = get_base_by_color(img[j][i][0])
                    if base_in_pos == '*' and ref_string[i] != '*':
                        base_in_pos = '.'
                    freq_dictionary[alt1+alt2][base_in_pos] += 1

    keys = ['AA', 'AC', 'AG', 'AT', 'CC', 'CG', 'CT', 'GG', 'GT', 'TT', '*A', '*C', '*G', '*T', '.A', '.C', '.G', '.T']
    result = [freq_dictionary[key] for key in keys]

    return result


def main(file_name, num_threads):
    global_freq_dict = defaultdict(lambda: defaultdict(int))
    arg_list = []

    with open(file_name, "r") as ins:
        for line in ins:
            line = line.rstrip()
            if not line:
                continue
            line = line.split(',')
            img_file = line[0]
            shape = (int(line[1]), int(line[2]), int(line[3]))
            interval_start, interval_end = int(line[4]), int(line[5])
            pos_vals = line[6]
            vcf_alt1 = line[7]
            vcf_alt2 = line[8]
            arg = (img_file, shape, vcf_alt1, vcf_alt2)
            # workers.append(pool.apply_async(analyze_it, args=(arg,)))

            arg_list.append(arg)

    list_arg_tuples = []
    for i in range(0, len(arg_list), 10):
        chunked_args = arg_list[i:i+10]
        list_arg_tuples.append(chunked_args)

    for i in tqdm(range(0, len(list_arg_tuples), num_threads)):
        start = i
        end = min(i+num_threads, len(list_arg_tuples))
        required_threads = end - start
        purged_arg_list = list_arg_tuples[start:end]
        pool = Pool(processes=required_threads)
        results = pool.map(analyze_it, purged_arg_list)
        # pool.join()
        for result in results:
            keys = ['AA', 'AC', 'AG', 'AT', 'CC', 'CG', 'CT', 'GG', 'GT', 'TT', '*A', '*C', '*G', '*T', '.A', '.C',
                    '.G', '.T']
            for i, base_dict in enumerate(result):
                for base in base_dict:
                    global_freq_dict[keys[i]][base] += base_dict[base]
        pool.close()
    return global_freq_dict


file_name = sys.argv[1]
num_threads = int(sys.argv[2])
frequency_dictionary = main(file_name, num_threads)


dictionary2 = OrderedDict(sorted(frequency_dictionary.items(), key=lambda t: t[0]))
fig = plt.figure()
fig.text(0.5, 0.02, 'Bases', ha='center', va='center', size=8)
fig.text(0.06, 0.5, 'Raw counts', ha='center', va='center', rotation='vertical', size=8)

plt.suptitle("Frequency distribution per class", size=8)
plts_in_y = 2
import math
plts_in_x = int(math.ceil(len(frequency_dictionary.items()) / plts_in_y))
subplot_val = plts_in_x * 100 + plts_in_y * 10

for i, key in enumerate(dictionary2):
    base_keys = ['A', 'C', 'G', 'T', '*', '.']
    value_list = [dictionary2[key][base] for base in base_keys]
    print(key, value_list)

for i, key in enumerate(dictionary2):
    base_keys = ['A', 'C', 'G', 'T', '*', '.']
    value_list = [dictionary2[key][base] for base in base_keys]
    plt.subplot(subplot_val+i+1)
    plt.title(key, fontsize=8)
    plt.bar(range(len(value_list)), value_list)
    plt.xticks(range(len(base_keys)), ['A', 'C', 'G', 'T', '*', '.'], fontsize=6)
    plt.yticks([], [])
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
# plt.show()
plt.savefig(file_name.split('/')[-1].split('.')[0]+"_Visualized.png", dpi=400)