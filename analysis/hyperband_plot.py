import pickle
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def plot_hyperband(pkl_file_path):
    hyperband_results = pickle.load(open(pkl_file_path, "rb"))
    result_dict_hash = defaultdict(list)
    result_reverse_hash = defaultdict(list)
    param_id = 1
    results_xy = defaultdict(list)
    for result in hyperband_results:
        params = tuple(result['params'].values())
        if params not in result_dict_hash.keys():
            result_dict_hash[params] = param_id
            result_reverse_hash[param_id] = params
            param_id += 1
        results_xy[result_dict_hash[params]].append((result['iterations'], result['loss']))

    labels = []
    import operator
    results_xy = sorted(results_xy.items(), key=operator.itemgetter(1))

    for i in range(len(results_xy)):
        print(results_xy[i][1])
        plt.plot(*zip(*results_xy[i][1]))
        indx = results_xy[i][0]
        print(result_reverse_hash[indx])
        labels.append(r'$(%f ,%f)$' % (result_reverse_hash[indx][0], result_reverse_hash[indx][1]))
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0.00002, 0.00016))
    plt.legend(labels, ncol=4, loc='upper center',
               bbox_to_anchor=[0.5, 1.1],
               columnspacing=1.0, labelspacing=0.0,
               handletextpad=0.0, handlelength=1.5,
               fancybox=True, shadow=True)
        # labels.append(r'$y = %ix + %i$' % (i, 5 * i))
    plt.show()


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks.
    '''
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--hyperband_result",
        type=str,
        required=True,
        help="Training data description csv file."
    )

    FLAGS, unparsed = parser.parse_known_args()

    plot_hyperband(FLAGS.hyperband_result)