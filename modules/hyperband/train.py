from __future__ import print_function
import sys
import torch
import torch.nn as nn
import random
import os
from torchvision import transforms
from tqdm import tqdm
import numpy as np
# Custom generator for our dataset
from torch.utils.data import DataLoader
from modules.core.dataloader import SequenceDataset
from modules.handlers.TextColor import TextColor
from modules.models.ModelHandler import ModelHandler
from modules.hyperband.test import test
from modules.core.ImageGenerator import CONTEXT_SIZE
"""
Train a model and return the model and optimizer trained.

Input:
- A train CSV containing training image set information (usually chr1-18)

Return:
- A trained model
"""
FLANK_SIZE = 10
CLASS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def train(train_file, test_file, batch_size, epoch_limit, prev_ite, gpu_mode, num_workers, retrain_model,
          retrain_model_path, gru_layers, hidden_size, encoder_lr, encoder_decay, decoder_lr, decoder_decay):
    """
    Train a model and save
    :param train_file: A CSV file containing train image information
    :param batch_size: Batch size for training
    :param epoch_limit: Number of epochs to train on
    :param gpu_mode: If true the model will be trained on GPU
    :param num_workers: Number of workers for data loading
    :param retrain_model: If true then a trained model will be retrained
    :param retrain_model_path: If retrain model is true then the model will be loaded from here
    :param hidden_size: Size of hidden GRU units
    :param encoder_lr: Encoder model's learning rate
    :param encoder_decay: Encoder model's weight decay
    :param decoder_lr: Decoder model's learning rate
    :param decoder_decay: Decoder model's weight decay
    :return:
    """
    transformations = transforms.Compose([transforms.ToTensor()])

    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)
    train_data_set = SequenceDataset(train_file, transformations)
    train_loader = DataLoader(train_data_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=gpu_mode
                              )
    # this needs to change
    encoder_model, decoder_model = ModelHandler.get_new_model(input_channels=10,
                                                              gru_layers=gru_layers,
                                                              hidden_size=hidden_size,
                                                              num_classes=6)
    encoder_optimizer = torch.optim.Adam(encoder_model.parameters(), lr=encoder_lr,
                                         weight_decay=encoder_decay)
    decoder_optimizer = torch.optim.Adam(decoder_model.parameters(), lr=decoder_lr,
                                         weight_decay=decoder_decay)
    if retrain_model is True:
        if os.path.isfile(retrain_model_path) is False:
            sys.stderr.write(TextColor.RED + "ERROR: INVALID PATH TO RETRAIN PATH MODEL --retrain_model_path\n")
            exit(1)
        sys.stderr.write(TextColor.GREEN + "INFO: RETRAIN MODEL LOADING\n" + TextColor.END)
        encoder_model, decoder_model = ModelHandler.load_model_for_training(encoder_model,
                                                                            decoder_model,
                                                                            retrain_model_path)

        encoder_optimizer, decoder_optimizer = ModelHandler.load_optimizer(encoder_optimizer,
                                                                           decoder_optimizer,
                                                                           retrain_model_path,
                                                                           gpu_mode)
        sys.stderr.write(TextColor.GREEN + "INFO: RETRAIN MODEL LOADED\n" + TextColor.END)

    if gpu_mode:
        encoder_model = torch.nn.DataParallel(encoder_model).cuda()
        decoder_model = torch.nn.DataParallel(decoder_model).cuda()

    class_weights = torch.FloatTensor(CLASS_WEIGHTS)
    # Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if gpu_mode is True:
        criterion = criterion.cuda()

    start_epoch = prev_ite

    # Train the Model
    sys.stderr.write(TextColor.PURPLE + 'Training starting\n' + TextColor.END)
    stats = dict()
    stats['loss_epoch'] = []
    stats['accuracy_epoch'] = []

    for epoch in range(start_epoch, epoch_limit, 1):
        total_loss = 0
        total_images = 0
        sys.stderr.write(TextColor.BLUE + 'Train epoch: ' + str(epoch + 1) + "\n")
        # make sure the model is in train mode. BN is different in train and eval.
        encoder_model.train()
        decoder_model.train()

        with tqdm(total=len(train_loader), desc='Loss', leave=True, dynamic_ncols=True) as progress_bar:
            for images, labels in train_loader:
                if gpu_mode:
                    # encoder_hidden = encoder_hidden.cuda()
                    images = images.cuda()
                    labels = labels.cuda()

                encoder_hidden = torch.FloatTensor(labels.size(0), gru_layers * 2, hidden_size).zero_()

                # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
                if gpu_mode:
                    encoder_hidden = encoder_hidden.cuda()

                window_size = images.size(2) - 2 * CONTEXT_SIZE
                index_start = CONTEXT_SIZE
                end_index = index_start + window_size
                current_batch_size = images.size(0)
                total_seq_length = window_size
                loss = 0
                output_enc, hidden_dec = encoder_model(images, encoder_hidden)
                for seq_index in range(index_start, end_index):
                    attention_index = torch.from_numpy(np.asarray([seq_index-index_start] * current_batch_size)).view(-1, 1)

                    attention_index_onehot = torch.FloatTensor(current_batch_size, total_seq_length)

                    attention_index_onehot.zero_()
                    attention_index_onehot.scatter_(1, attention_index, 1)

                    y = labels[:, seq_index - index_start]

                    output_dec, decoder_hidden, attn = decoder_model(attention_index_onehot, output_enc, hidden_dec)

                    loss += criterion(output_dec, y)

                    del attn

                total_loss += loss.item()
                total_images += labels.size(0)

                # loss + optimize
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

                # update the progress bar
                avg_loss = (total_loss / total_images) if total_images else 0
                progress_bar.set_description("Loss: " + str(avg_loss))
                progress_bar.refresh()
                progress_bar.update(1)

                del images, labels, encoder_hidden
            progress_bar.close()

        stats_dictioanry = test(test_file, batch_size, gpu_mode, encoder_model, decoder_model, num_workers,
                                gru_layers, hidden_size, num_classes=6)
        stats['loss'] = stats_dictioanry['loss']
        stats['accuracy'] = stats_dictioanry['accuracy']
        stats['loss_epoch'].append((epoch, stats_dictioanry['loss']))
        stats['accuracy_epoch'].append((epoch, stats_dictioanry['accuracy']))

    sys.stderr.write(TextColor.PURPLE + 'Finished training\n' + TextColor.END)

    return encoder_model, decoder_model, encoder_optimizer, decoder_optimizer, stats

