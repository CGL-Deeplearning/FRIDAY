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
from torch.autograd import Variable
from modules.core.dataloader import SequenceDataset
from modules.models.ModelHandler import ModelHandler
from modules.models.Seq2Seq_atn import EncoderCRNN, AttnDecoderRNN
from modules.handlers.TextColor import TextColor
"""
Train a model and save the model that performs best.

Input:
- A train CSV containing training image set information (usually chr1-18)
- A test CSV containing testing image set information (usually chr19)

Output:
- A trained model
"""
FLANK_SIZE = 10
WINDOW_SIZE = 1


def test(data_file, batch_size, gpu_mode, encoder_model, decoder_model, num_classes, num_workers):
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
    encoder_model = encoder_model.eval()
    decoder_model = decoder_model.eval()
    if gpu_mode:
        encoder_model = encoder_model.cuda()
        decoder_model = decoder_model.cuda()

    # Loss
    test_criterion = nn.CrossEntropyLoss()

    # Test the Model
    # sys.stderr.write(TextColor.PURPLE + 'Test starting\n' + TextColor.END)
    confusion_matrix = meter.ConfusionMeter(num_classes)

    total_loss = 0
    total_images = 0
    with tqdm(total=len(test_loader), desc='Accuracy: ', leave=True, dynamic_ncols=True) as pbar:
        for i, (images, labels, positional_information) in enumerate(test_loader):
            images = Variable(images)
            labels = Variable(labels)
            if gpu_mode:
                # encoder_hidden = encoder_hidden.cuda()
                images = images.cuda()
                labels = labels.cuda()

            decoder_input = Variable(torch.LongTensor(labels.size(0), 1).zero_())
            if gpu_mode:
                decoder_input = decoder_input.cuda()

            window_size = images.size(2) - 2 * FLANK_SIZE
            index_start = FLANK_SIZE
            end_index = index_start + window_size
            for seq_index in range(index_start, end_index):
                # get the logits
                x = images[:, :, seq_index - FLANK_SIZE:seq_index + FLANK_SIZE + 1, :]
                y = labels[:, seq_index - index_start]

                encoder_output, encoder_hidden = encoder_model(x)

                outputs, hidden = decoder_model(decoder_input, encoder_hidden, encoder_output)
                topv, topi = outputs.squeeze().topk(1)
                decoder_input = topi.detach()  # detach from history as input

                # loss
                loss = test_criterion(outputs, y)
                total_loss += loss.data[0]
                total_images += labels.size(0)
                confusion_matrix.add(outputs.data.contiguous().view(-1, num_classes), y.data.contiguous().view(-1))

            pbar.update(1)
            cm_value = confusion_matrix.value()
            accuracy = 100.0 * (cm_value[1][1] + cm_value[2][2] + cm_value[3][3] + cm_value[4][4] +
                                cm_value[5][5]) / (cm_value.sum() - cm_value[0][0])
            pbar.set_description("Accuracy: " + str(accuracy))

    avg_loss = total_loss / total_images if total_images else 0
    # print('Test Loss: ' + str(avg_loss))
    # print('Confusion Matrix: \n', confusion_matrix.conf)

    sys.stderr.write(TextColor.YELLOW+'Test Loss: ' + str(avg_loss) + "\n"+TextColor.END)
    sys.stderr.write("Confusion Matrix: \n" + str(confusion_matrix.conf) + "\n" + TextColor.END)


def train(train_file, validation_file, batch_size, epoch_limit, gpu_mode, num_workers, file_name, num_classes=6):
    """
    Train a model and save
    :param train_file: A CSV file containing train image information
    :param validation_file: A CSV file containing test image information
    :param batch_size: Batch size for training
    :param epoch_limit: Number of epochs to train on
    :param gpu_mode: If true the model will be trained on GPU
    :param num_workers: Number of workers for data loading
    :param num_classes: Number of output classes
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

    encoder_model = EncoderCRNN(image_channels=8, hidden_size=256)
    decoder_model = AttnDecoderRNN(hidden_size=512, num_classes=6, max_length=1)

    encoder_optimizer = torch.optim.Adam(encoder_model.parameters(), lr=0.000217) # 0.000217
    decoder_optimizer = torch.optim.Adam(decoder_model.parameters(), lr=0.000217)

    if gpu_mode is True:
        encoder_model = encoder_model.cuda()
        decoder_model = decoder_model.cuda()

    # Loss function
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0

    if gpu_mode:
        encoder_model = torch.nn.DataParallel(encoder_model).cuda()
        decoder_model = torch.nn.DataParallel(decoder_model).cuda()

    # Train the Model
    sys.stderr.write(TextColor.PURPLE + 'Training starting\n' + TextColor.END)
    for epoch in range(start_epoch, epoch_limit, 1):
        total_loss = 0
        total_images = 0
        sys.stderr.write(TextColor.BLUE + 'Train epoch: ' + str(epoch + 1) + "\n")
        with tqdm(total=len(train_loader), desc='Loss', leave=True, dynamic_ncols=True) as progress_bar:
            for images, labels, positional_information in train_loader:
                images = Variable(images)
                labels = Variable(labels)
                if gpu_mode:
                    # encoder_hidden = encoder_hidden.cuda()
                    images = images.cuda()
                    labels = labels.cuda()

                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                teacher_forcing_ratio = 0.5
                decoder_input = Variable(torch.LongTensor(labels.size(0), 1).zero_())
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

                if gpu_mode:
                    decoder_input = decoder_input.cuda()

                window_size = images.size(2) - 2 * FLANK_SIZE
                index_start = FLANK_SIZE
                end_index = index_start + window_size
                for seq_index in range(index_start, end_index):

                    x = images[:, :, seq_index-FLANK_SIZE:seq_index+FLANK_SIZE+1, :]
                    y = labels[:, seq_index - index_start]

                    encoder_output, encoder_hidden = encoder_model(x)

                    print("In Train loop: ", decoder_input.size(), encoder_hidden.size(), encoder_output.size())
                    outputs, hidden = decoder_model(decoder_input, encoder_hidden, encoder_output)
                    # loss + optimize
                    loss = criterion(outputs, y)
                    loss.backward()
                    encoder_optimizer.step()
                    decoder_optimizer.step()
                    total_loss += loss.data[0]
                    total_images += labels.size(0)

                    if use_teacher_forcing:
                        decoder_input = y.unsqueeze(1)
                    else:
                        topv, topi = outputs.squeeze().topk(1)
                        decoder_input = topi.detach()  # detach from history as input

                # update the progress bar
                avg_loss = (total_loss / total_images) if total_images else 0
                progress_bar.set_description("Loss: " + str(avg_loss))
                progress_bar.refresh()
                progress_bar.update(1)
            progress_bar.close()

        # After each epoch do validation
        test(validation_file, batch_size, gpu_mode, encoder_model, decoder_model, num_classes, num_workers)
        encoder_model = encoder_model.train()
        decoder_model = decoder_model.train()
        save_best_model(encoder_model, encoder_optimizer, file_name+"_epoch_"+str(epoch+1))

    sys.stderr.write(TextColor.PURPLE + 'Finished training\n' + TextColor.END)


def save_best_model(best_model, optimizer, file_name):
    """
    Save the best model
    :param best_model: A trained model
    :param optimizer: Optimizer
    :param file_name: Output file name
    :return:
    """
    # sys.stderr.write(TextColor.BLUE + "SAVING MODEL.\n" + TextColor.END)
    if os.path.isfile(file_name + '_model.pkl'):
        os.remove(file_name + '_model.pkl')
    if os.path.isfile(file_name + '_checkpoint.pkl'):
        os.remove(file_name + '_checkpoint.pkl')
    torch.save(best_model, file_name + '_model.pkl')
    ModelHandler.save_checkpoint({
        'state_dict': best_model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, file_name + '_checkpoint.pkl')
    sys.stderr.write(TextColor.RED + "MODEL SAVED SUCCESSFULLY.\n" + TextColor.END)


def handle_output_directory(output_dir):
    """
    Process the output directory and return a valid directory where we save the output
    :param output_dir: Output directory path
    :return:
    """
    timestr = time.strftime("%m%d%Y_%H%M%S")
    # process the output directory
    if output_dir[-1] != "/":
        output_dir += "/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # create an internal directory so we don't overwrite previous runs
    model_save_dir = output_dir + "trained_models_" + timestr + "/"
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    stats_directory = model_save_dir + "stats_" + timestr + "/"

    if not os.path.exists(stats_directory):
        os.mkdir(stats_directory)

    return model_save_dir, stats_directory


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Training data description csv file."
    )
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
        "--epoch_size",
        type=int,
        required=False,
        default=10,
        help="Epoch size for training iteration."
    )
    parser.add_argument(
        "--model_out",
        type=str,
        required=False,
        default='./model',
        help="Path and file_name to save model, default is ./model"
    )
    parser.add_argument(
        "--retrain_model",
        type=bool,
        default=False,
        help="If true then retrain a pre-trained mode."
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
    model_dir, stats_dir = handle_output_directory(FLAGS.model_out.rpartition('/')[0]+"/")
    model_out_dir = model_dir + "seq2seq_model_"
    sys.stderr.write(TextColor.BLUE + "THE MODEL AND STATS LOCATION: " + str(model_dir) + "\n" + TextColor.END)

    train(FLAGS.train_file, FLAGS.test_file, FLAGS.batch_size, FLAGS.epoch_size, FLAGS.gpu_mode, FLAGS.num_workers,
          model_out_dir)
