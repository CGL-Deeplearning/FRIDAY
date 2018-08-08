import argparse
import os
import sys

import torch
import torch.nn.parallel
from modules.models.test import test
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


def do_test(test_file, batch_size, gpu_mode, num_workers, model_path, num_classes=6):
    """
    Train a model and save
    :param test_file: A CSV file containing test image information
    :param batch_size: Batch size for training
    :param gpu_mode: If true the model will be trained on GPU
    :param num_workers: Number of workers for data loading
    :param model_path: Path to a saved model
    :param num_classes: Number of output classes
    :return:
    """
    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)

    # this needs to change
    hidden_size = 256
    gru_layers = 1
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
        test(test_file, batch_size, gru_layers, hidden_size, gpu_mode, encoder_model, decoder_model, num_classes,
             num_workers)

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
