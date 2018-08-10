import matplotlib
matplotlib.use('Agg')
import sys
import torch
from tqdm import tqdm
import torchnet.meter as meter
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from modules.core.dataloader import SequenceDataset
from modules.handlers.TextColor import TextColor
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns



"""
This script will evaluate a model and return the loss value.

Input:
- A trained model
- A test CSV file to evaluate

Returns:
- Loss value
"""
FLANK_SIZE = 10
CLASS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
sns.set(color_codes=True)
sns.set_palette(sns.color_palette("colorblind"))
sns.set_style("white")


def test(data_file, batch_size, gpu_mode, encoder_model, decoder_model, num_workers, gru_layers, hidden_size, num_classes=6):
    transformations = transforms.Compose([transforms.ToTensor()])

    # data loader
    test_data = SequenceDataset(data_file, transformations)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=gpu_mode)
    sys.stderr.write(TextColor.CYAN + 'Test data loaded\n')

    # set the evaluation mode of the model
    encoder_model.eval()
    decoder_model.eval()

    class_weights = torch.FloatTensor(CLASS_WEIGHTS)
    # Loss
    test_criterion = nn.CrossEntropyLoss(weight=class_weights)

    if gpu_mode is True:
        test_criterion = test_criterion.cuda()

    # Test the Model
    # sys.stderr.write(TextColor.PURPLE + 'Test starting\n' + TextColor.END)
    confusion_matrix = meter.ConfusionMeter(num_classes)

    total_loss = 0
    total_images = 0
    accuracy = 0
    correct_predictions = defaultdict(int)
    incorrect_predictions = defaultdict(int)
    total_cases = defaultdict(int)
    positional_class_distribution = defaultdict(int)
    positional_hom_cases = defaultdict(int)
    total_incorrect_calls = 0
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Accuracy: ', leave=True, ncols=100) as pbar:
            for i, (images, labels) in enumerate(test_loader):
                if gpu_mode:
                    # encoder_hidden = encoder_hidden.cuda()
                    images = images.cuda()
                    labels = labels.cuda()

                encoder_hidden = torch.FloatTensor(labels.size(0), gru_layers * 2, hidden_size).zero_()

                if gpu_mode:
                    encoder_hidden = encoder_hidden.cuda()

                decoder_attentions = torch.zeros(images.size(0), images.size(2), 10)

                context_vector, hidden_encoder = encoder_model(images, encoder_hidden)
                loss = 0
                seq_length = images.size(2)
                for seq_index in range(0, seq_length):
                    current_batch_size = images.size(0)
                    y = labels[:, seq_index]
                    attention_index = torch.from_numpy(np.asarray([seq_index] * current_batch_size)).view(-1, 1)

                    attention_index_onehot = torch.FloatTensor(current_batch_size, seq_length)

                    attention_index_onehot.zero_()
                    attention_index_onehot.scatter_(1, attention_index, 1)

                    output_dec, decoder_hidden, attn = decoder_model(attention_index_onehot,
                                                                     context_vector=context_vector,
                                                                     encoder_hidden=hidden_encoder)

                    decoder_attentions[:, seq_index] = attn.view(attn.size(0), -1).data

                    # loss + optimize
                    loss += test_criterion(output_dec, y)
                    confusion_matrix.add(output_dec.data.contiguous().view(-1, num_classes),
                                         y.data.contiguous().view(-1))

                    for batch in range(current_batch_size):
                        true_label = y[batch].item()

                        m = nn.Softmax(dim=1)
                        soft_probs = m(output_dec)
                        output_preds = soft_probs.cpu()
                        preds = output_preds[batch, :].data
                        top_n, top_i = preds.topk(1)
                        predicted_label = top_i[0].item()

                        if true_label != 0:
                            positional_class_distribution[seq_index] += 1
                        else:
                            positional_hom_cases[seq_index] += 1

                        if true_label != 0 and predicted_label == true_label:
                            correct_predictions[seq_index] += 1
                        elif predicted_label != true_label:
                            incorrect_predictions[seq_index] += 1
                            total_incorrect_calls += 1
                        total_cases[seq_index] += 1

                total_loss += loss.item()
                total_images += labels.size(0)

                # print(labels.data.cpu().numpy())
                # plt.matshow(decoder_attentions.numpy()[0])
                # plt.title(str(labels.data.cpu().numpy()[0]), fontsize=12)
                # plt.show()
                # exit()
                # plt.savefig('/data/users/kishwar/train_data/plots/'+"plot_" + str(i) + ".png")
                # exit()

                pbar.update(1)
                cm_value = confusion_matrix.value()
                denom = (cm_value.sum() - cm_value[0][0]) if (cm_value.sum() - cm_value[0][0]) > 0 else 1.0
                accuracy = 100.0 * (cm_value[1][1] + cm_value[2][2] + cm_value[3][3] + cm_value[4][4] +
                                    cm_value[5][5]) / denom
                pbar.set_description("Accuracy: " + str(accuracy))

    x_axes = []
    y_axes_accuracy = []
    y_axes_class = []
    y_axes_hom = []
    for i in range(0, 10):
        x = i
        accuracy = incorrect_predictions[i] / total_incorrect_calls if total_incorrect_calls else 0
        non_hom_cases = positional_class_distribution[i] / total_cases[i] if total_cases[i] else 0
        hom_cases = positional_hom_cases[i] / total_cases[i] if total_cases[i] else 0

        x_axes.append(x)
        y_axes_accuracy.append(accuracy)
        y_axes_class.append(non_hom_cases)
        y_axes_hom.append(hom_cases)

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 7), sharey=False, frameon=True)
    ax_ = axes[0]
    ax_.bar(x_axes, y_axes_accuracy)
    ax_.set_xticks(np.arange(0, 10, step=1))
    ax_.set_yticks(np.arange(0, 1.1, step=0.1))
    ax_.set_ylim(0, 1.1)
    ax_.set_xlabel('Position')
    ax_.set_ylabel('Wrong prediction distribution')
    ax_.set_title('Position-wise wrong prediction distribution', fontsize=12, fontweight='bold')

    ax_ = axes[1]
    p1 = ax_.bar(x_axes, y_axes_hom)
    p2 = ax_.bar(x_axes, y_axes_class, bottom=y_axes_hom)
    ax_.set_xticks(np.arange(0, 10, step=1))
    ax_.set_yticks(np.arange(0, 1.1, step=0.1))
    ax_.set_ylim(0, 1.1)
    ax_.set_xlabel('Position')
    ax_.set_ylabel('Non-homozygous class frequency')
    ax_.set_title('Position-wise class distribution', fontsize=12, fontweight='bold')
    ax_.legend((p1[0], p2[0]), ('Hom', 'Non-Hom'), loc='upper center', bbox_to_anchor=(0.5, 1.0),
               fancybox=True, shadow=True, ncol=2)

    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(top=0.88)
    plt.savefig('wrong_prediction_distribution.png', dpi=400)

    avg_loss = total_loss / total_images if total_images else 0

    sys.stderr.write(TextColor.YELLOW+'\nTest Loss: ' + str(avg_loss) + "\n"+TextColor.END)
    sys.stderr.write("Confusion Matrix: \n" + str(confusion_matrix.conf) + "\n" + TextColor.END)

    return {'loss': avg_loss, 'accuracy': accuracy, 'confusion_matrix': str(confusion_matrix.conf)}
