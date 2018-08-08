from __future__ import print_function
import sys
import torch
import torch.nn as nn
import random
import os
from torchvision import transforms
from tqdm import tqdm

# Custom generator for our dataset
from torch.utils.data import DataLoader
from modules.core.dataloader import SequenceDataset
from modules.handlers.TextColor import TextColor
from modules.models.ModelHandler import ModelHandler
from modules.models.test import test
"""
Train a model and return the model and optimizer trained.

Input:
- A train CSV containing training image set information (usually chr1-18)

Return:
- A trained model
"""
FLANK_SIZE = 10
CLASS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def save_best_model(encoder_model, decoder_model, encoder_optimizer, decoder_optimizer, hidden_size, layers, epoch,
                    file_name):
    """
    Save the best model
    :param encoder_model: A trained encoder model
    :param decoder_model: A trained decoder model
    :param encoder_optimizer: Encoder optimizer
    :param decoder_optimizer: Decoder optimizer
    :param file_name: Output file name
    :return:
    """
    if os.path.isfile(file_name + '_checkpoint.pkl'):
        os.remove(file_name + '_checkpoint.pkl')
    ModelHandler.save_checkpoint({
        'encoder_state_dict': encoder_model.state_dict(),
        'decoder_state_dict': decoder_model.state_dict(),
        'encoder_optimizer': encoder_optimizer.state_dict(),
        'decoder_optimizer': decoder_optimizer.state_dict(),
        'hidden_size': hidden_size,
        'gru_layers': layers,
        'epochs': epoch,
    }, file_name + '_checkpoint.pkl')
    sys.stderr.write(TextColor.RED + "\nMODEL SAVED SUCCESSFULLY.\n" + TextColor.END)


def train(train_file, test_file, batch_size, epoch_limit, gpu_mode, num_workers, retrain_model,
          retrain_model_path, gru_layers, hidden_size, encoder_lr, encoder_decay, decoder_lr, decoder_decay,
          model_dir, stats_dir, train_mode):

    if train_mode is True:
        train_loss_logger = open(stats_dir + "train_loss.csv", 'w')
        test_loss_logger = open(stats_dir + "test_loss.csv", 'w')
        confusion_matrix_logger = open(stats_dir + "confusion_matrix.txt", 'w')
    else:
        train_loss_logger = None
        test_loss_logger = None
        confusion_matrix_logger = None

    transformations = transforms.Compose([transforms.ToTensor()])

    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)
    train_data_set = SequenceDataset(train_file, transformations)
    train_loader = DataLoader(train_data_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=gpu_mode
                              )

    if retrain_model is True:
        if os.path.isfile(retrain_model_path) is False:
            sys.stderr.write(TextColor.RED + "ERROR: INVALID PATH TO RETRAIN PATH MODEL --retrain_model_path\n")
            exit(1)
        sys.stderr.write(TextColor.GREEN + "INFO: RETRAIN MODEL LOADING\n" + TextColor.END)
        encoder_model, decoder_model, hidden_size, gru_layers, prev_ite = \
            ModelHandler.load_model_for_training(retrain_model_path, input_channels=10, num_classes=6)

        if train_mode is True:
            epoch_limit = prev_ite + epoch_limit

        sys.stderr.write(TextColor.GREEN + "INFO: RETRAIN MODEL LOADED\n" + TextColor.END)
    else:
        encoder_model, decoder_model = ModelHandler.get_new_model(input_channels=10,
                                                                  gru_layers=gru_layers,
                                                                  hidden_size=hidden_size,
                                                                  num_classes=6)
        prev_ite = 0

    encoder_optimizer = torch.optim.Adam(encoder_model.parameters(), lr=encoder_lr,
                                         weight_decay=encoder_decay)
    decoder_optimizer = torch.optim.Adam(decoder_model.parameters(), lr=decoder_lr,
                                         weight_decay=decoder_decay)

    if retrain_model is True:
        sys.stderr.write(TextColor.GREEN + "INFO: OPTIMIZER LOADING\n" + TextColor.END)
        encoder_optimizer, decoder_optimizer = ModelHandler.load_optimizer(encoder_optimizer, decoder_optimizer,
                                                                           retrain_model_path, gpu_mode)
        sys.stderr.write(TextColor.GREEN + "INFO: OPTIMIZER LOADED\n" + TextColor.END)

    if gpu_mode:
        encoder_model = torch.nn.DistributedDataParallel(encoder_model).cuda()
        decoder_model = torch.nn.DistributedDataParallel(decoder_model).cuda()

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
    sys.stderr.write(TextColor.PURPLE + 'Start: ' + str(start_epoch + 1) + ' End: ' + str(epoch_limit + 1) + "EPOCH\n")
    for epoch in range(start_epoch, epoch_limit, 1):
        total_loss = 0
        total_images = 0
        sys.stderr.write(TextColor.BLUE + 'Train epoch: ' + str(epoch + 1) + "\n")
        # make sure the model is in train mode. BN is different in train and eval.
        encoder_model.train()
        decoder_model.train()
        batch_no = 1
        with tqdm(total=len(train_loader), desc='Loss', leave=True, dynamic_ncols=True) as progress_bar:
            for images, labels in train_loader:
                if gpu_mode:
                    # encoder_hidden = encoder_hidden.cuda()
                    images = images.cuda()
                    labels = labels.cuda()

                teacher_forcing_ratio = 0.0
                decoder_input = torch.LongTensor(labels.size(0), 1).zero_()
                encoder_hidden = torch.FloatTensor(labels.size(0), gru_layers * 2, hidden_size).zero_()

                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
                if gpu_mode:
                    decoder_input = decoder_input.cuda()
                    encoder_hidden = encoder_hidden.cuda()

                window_size = images.size(2) - 2 * FLANK_SIZE
                index_start = FLANK_SIZE
                end_index = index_start + int(window_size / 2) + 1

                for seq_index in range(index_start, end_index):
                    encoder_optimizer.zero_grad()
                    decoder_optimizer.zero_grad()

                    x = images[:, :, seq_index - FLANK_SIZE:seq_index + FLANK_SIZE + 1, :]
                    y = labels[:, seq_index - index_start]

                    output_enc, hidden_dec = encoder_model(x, encoder_hidden)
                    output_dec, decoder_hidden, attn = decoder_model(decoder_input, output_enc, hidden_dec)

                    encoder_hidden = decoder_hidden.detach()
                    if seq_index == 15:
                        # loss + optimize
                        loss = criterion(output_dec, y)
                        loss.backward()
                        encoder_optimizer.step()
                        decoder_optimizer.step()
                        total_loss += loss.item()
                        total_images += labels.size(0)

                    if use_teacher_forcing:
                        decoder_input = y
                    else:
                        topv, topi = output_dec.topk(1)
                        decoder_input = topi.squeeze().detach()

                # update the progress bar
                avg_loss = (total_loss / total_images) if total_images else 0
                progress_bar.set_description("Loss: " + str(avg_loss))
                if train_mode is True:
                    train_loss_logger.write(str(epoch + 1) + "," + str(batch_no) + "," + str(avg_loss) + "\n")
                progress_bar.refresh()
                progress_bar.update(1)
                batch_no += 1

                del decoder_input, encoder_hidden
            progress_bar.close()

        stats_dictioanry = test(test_file, batch_size, gpu_mode, encoder_model, decoder_model, num_workers,
                                gru_layers, hidden_size, num_classes=6)
        stats['loss'] = stats_dictioanry['loss']
        stats['accuracy'] = stats_dictioanry['accuracy']
        stats['loss_epoch'].append((epoch, stats_dictioanry['loss']))
        stats['accuracy_epoch'].append((epoch, stats_dictioanry['accuracy']))

        # update the loggers
        if train_mode is True:
            # save the model after each epoch
            save_best_model(encoder_model, decoder_model, encoder_optimizer, decoder_optimizer,
                            model_dir + "_epoch_" + str(epoch + 1))

            test_loss_logger.write(str(epoch + 1) + "," + str(stats['loss']) + "," + str(stats['accuracy']) + "\n")
            confusion_matrix_logger.write(str(epoch + 1) + "\n" + str(stats['confusion_matrix']) + "\n")
            train_loss_logger.flush()
            test_loss_logger.flush()
            confusion_matrix_logger.flush()

    sys.stderr.write(TextColor.PURPLE + 'Finished training\n' + TextColor.END)

    return encoder_model, decoder_model, encoder_optimizer, decoder_optimizer, stats

