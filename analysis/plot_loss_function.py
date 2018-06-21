import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from collections import OrderedDict


sns.set(color_codes=True)
stat_directory = sys.argv[1]
train_loss_file = stat_directory + "train_loss.csv"
test_loss_file = stat_directory + "test_loss.csv"

train_losses = []
train_x = []
test_losses = []
test_accuracy = []
test_x = []
epoch = 0
batch_size = 0
count = 1
with open(train_loss_file, "r") as ins:
    for line in ins:
        line = line.rstrip()
        if not line:
            continue
        epoch_, batch_size_, loss = line.split(',')
        train_losses.append(float(loss))
        train_x.append(count)
        batch_size = max(batch_size, int(batch_size_))
        epoch = max(epoch, int(epoch_))
        count += 1

with open(test_loss_file, "r") as ins:
    for line in ins:
        line = line.rstrip()
        if not line:
            continue
        epoch_, loss, accuracy = line.split(',')
        test_losses.append(float(loss))
        test_x.append((int(epoch_) * batch_size) + 1)
        test_accuracy.append(float(accuracy))

train, = plt.plot(train_x, train_losses)
test, = plt.plot(test_x, test_losses, 'ro-')
plt.legend([train, test], ['Train loss', 'Test loss'])

# fig, ax = plt.subplots()

# for i, v in enumerate(test_accuracy):
#     ax.text(i * batch_size, v, str(round(v, 2)) + "%",
#             fontweight='bold', ha='center', fontsize=8)

gap_in_ticks = 1
x_ticks = ()
for i in range(0, epoch+1, gap_in_ticks):
    if 0 < i < len(test_accuracy) + 1:
        accuracy = round(test_accuracy[i-1], 2)
    else:
        accuracy = 0.0
    x_ticks = x_ticks + (str(i) + "\n" + str(accuracy) + "%",)
# plt.ylim([0, 0.000004])
plt.xticks(range(1, batch_size * (epoch+1), batch_size*gap_in_ticks), x_ticks)
plt.xlabel('Epoch')
plt.ylabel('Loss')

# plt.title('CNN Training chr1~19')

plt.savefig(stat_directory.split('/')[-2] + '_loss' + '.png', dpi=400)
