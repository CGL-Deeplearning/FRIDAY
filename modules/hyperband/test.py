from __future__ import print_function
import sys
import torch
from tqdm import tqdm
import torchnet.meter as meter
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from modules.core.dataloader import SequenceDataset
from modules.handlers.TextColor import TextColor
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


def test(data_file, batch_size, gpu_mode, encoder_model, decoder_model, num_workers, hidden_size, num_classes=6):
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
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Accuracy: ', leave=True, dynamic_ncols=True) as pbar:
            for i, (images, labels, positional_information) in enumerate(test_loader):
                if gpu_mode:
                    # encoder_hidden = encoder_hidden.cuda()
                    images = images.cuda()
                    labels = labels.cuda()

                decoder_input = torch.LongTensor(labels.size(0), 1).zero_()
                encoder_hidden = torch.FloatTensor(labels.size(0), 2, hidden_size).zero_()

                if gpu_mode:
                    decoder_input = decoder_input.cuda()
                    encoder_hidden = encoder_hidden.cuda()

                window_size = images.size(2) - 2 * FLANK_SIZE
                index_start = FLANK_SIZE
                end_index = index_start + window_size

                for seq_index in range(index_start, end_index):
                    x = images[:, :, seq_index - FLANK_SIZE:seq_index + FLANK_SIZE + 1, :]
                    y = labels[:, seq_index - index_start]

                    output_enc, hidden_dec = encoder_model(x, encoder_hidden)
                    output_dec, decoder_hidden, attn = decoder_model(decoder_input, output_enc, hidden_dec)

                    encoder_hidden = decoder_hidden.detach()
                    topv, topi = output_dec.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                    # loss
                    loss = test_criterion(output_dec, y)
                    confusion_matrix.add(output_dec.data.contiguous().view(-1, num_classes), y.data.contiguous().view(-1))

                    total_loss += loss.item()
                    total_images += labels.size(0)

                pbar.update(1)
                cm_value = confusion_matrix.value()
                denom = (cm_value.sum() - cm_value[0][0]) if (cm_value.sum() - cm_value[0][0]) > 0 else 1.0
                accuracy = 100.0 * (cm_value[1][1] + cm_value[2][2] + cm_value[3][3] + cm_value[4][4] +
                                    cm_value[5][5]) / denom
                pbar.set_description("Accuracy: " + str(accuracy))

    avg_loss = total_loss / total_images if total_images else 0

    sys.stderr.write(TextColor.YELLOW+'\nTest Loss: ' + str(avg_loss) + "\n"+TextColor.END)
    sys.stderr.write("Confusion Matrix: \n" + str(confusion_matrix.conf) + "\n" + TextColor.END)

    return {'loss': avg_loss, 'accuracy': accuracy}
