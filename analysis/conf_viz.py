import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
array = \
[[1102600,    2794,      67,      41,       0,      15],
 [   2350,  317869,    1267,      15,       0,     217],
 [     66,     846,  162988,       0,       0,      84],
 [    101,      35,       0,     180,       0,      30],
 [      1,       0,      10,       8,       0,       7],
 [      8,     123,     114,       7,       0,    4995]]
array = np.array(array, dtype=float)
cm = np.array(array, dtype=float)
n_categories = 6
for i in range(n_categories):
    array[i] = array[i] / array[i].sum()

true_pos = np.diag(cm)
false_pos = np.sum(cm, axis=0) - true_pos
false_neg = np.sum(cm, axis=1) - true_pos
#print(true_pos)
#print(false_pos)
#print(false_neg)

precision = true_pos / (true_pos+false_pos)
recall = true_pos / (true_pos + false_neg)
F1 = 2 * precision * recall / (precision + recall)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1: ', F1)

sn.set(font_scale=3)
df_cm = pd.DataFrame(array, index=[i for i in ["0/0", "0/1", "1/1", "0/2", "2/2", "1/2"]],
                  columns=[i for i in ["0/0", "0/1", "1/1", "0/2", "2/2", "1/2"]])
plt.figure(figsize=(5*4, 4*4))

sn.heatmap(df_cm, annot=True)
plt.title('Confusion Matrix')
plt.savefig('Confusion_matrix')
