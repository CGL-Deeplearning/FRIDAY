import argparse
import os
import sys
import time
import random

import torch
import torch.nn.parallel
import torchnet.meter as meter
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from modules.core.dataloader import SequenceDataset
from modules.models.ModelHandler import ModelHandler
from modules.handlers.TextColor import TextColor
"""
FREEZE THIS BRANCH TO HAVE 1 WINDOW!!
Train a model and save the model that performs best.

Input:
- A train CSV containing training image set information (usually chr1-18)
- A test CSV containing testing image set information (usually chr19)

Output:
- A trained model
"""
FLANK_SIZE = 10
CLASS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test(data_file, batch_size, gru_layers, hidden_size, gpu_mode, encoder_model, decoder_model, num_classes, num_workers):
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
            for i, (images, labels) in enumerate(test_loader):
                if gpu_mode:
                    # encoder_hidden = encoder_hidden.cuda()
                    images = images.cuda()
                    labels = labels.cuda()

                decoder_input = torch.LongTensor(labels.size(0), 1).zero_()
                encoder_hidden = torch.FloatTensor(labels.size(0), gru_layers * 2, hidden_size).zero_()

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
                    del output_enc, hidden_dec, attn

                pbar.update(1)
                cm_value = confusion_matrix.value()
                denom = (cm_value.sum() - cm_value[0][0]) if (cm_value.sum() - cm_value[0][0]) > 0 else 1.0
                accuracy = 100.0 * (cm_value[1][1] + cm_value[2][2] + cm_value[3][3] + cm_value[4][4] +
                                    cm_value[5][5]) / denom
                pbar.set_description("Accuracy: " + str(accuracy))

                del images, labels, decoder_input, encoder_hidden

    avg_loss = total_loss / total_images if total_images else 0
    # print('Test Loss: ' + str(avg_loss))
    # print('Confusion Matrix: \n', confusion_matrix.conf)

    sys.stderr.write(TextColor.YELLOW+'\nTest Loss: ' + str(avg_loss) + "\n"+TextColor.END)
    sys.stderr.write("Confusion Matrix: \n" + str(confusion_matrix.conf) + "\n" + TextColor.END)

    return str(confusion_matrix.conf), avg_loss, accuracy


def do_test(test_file, batch_size, gpu_mode, num_workers, model_path, num_classes=6):
    """
    Train a model and save
    :param test_file: A CSV file containing test image information
    :param batch_size: Batch size for training
    :param gpu_mode: If true the model will be trained on GPU
    :param num_workers: Number of workers for data loading
    :param num_classes: Number of output classes
    :return:
    """
    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)

    # this needs to change
    hidden_size = 512
    gru_layers = 3
    encoder_model, decoder_model = ModelHandler.get_new_model(input_channels=10,
                                                              gru_layers=gru_layers,
                                                              hidden_size=hidden_size,
                                                              num_classes=6)

    if os.path.isfile(model_path) is False:
        sys.stderr.write(TextColor.RED + "ERROR: INVALID PATH TO MODEL\n")
        exit(1)

    sys.stderr.write(TextColor.GREEN + "INFO: MODEL LOADING\n" + TextColor.END)
    encoder_model, decoder_model = ModelHandler.load_model_for_training(encoder_model,
                                                                        decoder_model,
                                                                        model_path)

    sys.stderr.write(TextColor.GREEN + "INFO: MODEL LOADED\n" + TextColor.END)

    if gpu_mode:
        encoder_model = torch.nn.DataParallel(encoder_model).cuda()
        decoder_model = torch.nn.DataParallel(decoder_model).cuda()

    confusion_matrix, test_loss, accuracy = \
        test(test_file, batch_size, gru_layers, hidden_size, gpu_mode, encoder_model, decoder_model, num_classes, num_workers)

    sys.stderr.write(TextColor.PURPLE + 'DONE\n' + TextColor.END)


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Training data description csv file."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=100,
        help="Batch size for training, default is 100."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default='./model',
        help="Path of the model to load and retrain"
    )
    parser.add_argument(
        "--gpu_mode",
        type=bool,
        default=False,
        help="If true then cuda is on."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=40,
        help="Epoch size for training iteration."
    )
    FLAGS, not_parsed = parser.parse_known_args()
    do_test(FLAGS.test_file, FLAGS.batch_size, FLAGS.gpu_mode, FLAGS.num_workers, FLAGS.model_path)
