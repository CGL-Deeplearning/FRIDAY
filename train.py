import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import os
import sys
import time

import torch
import torch.nn.parallel
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from modules.core.dataloader import SequenceDataset
from modules.models.ModelHandler import ModelHandler
from modules.models.Seq2Seq_deepspeech import SimpleModel
from modules.handlers.TextColor import TextColor
"""
Train a model and save the model that performs best.

Input:
- A train CSV containing training image set information (usually chr1-18)
- A test CSV containing testing image set information (usually chr19)

Output:
- A trained model
"""
FLANK_SIZE = 2
WINDOW_SIZE = 20


def plot_confusion_matrix(num_classes, confusion, epoch_no):
    base_to_letter_map = {'0/0': 0, '0/1': 1, '1/1': 2, '0/2': 3, '2/2': 4, '1/2': 5}
    all_categories = list(sorted(base_to_letter_map, key=lambda k: base_to_letter_map[k]))
    # Normalize by dividing every row by its sum
    for i in range(num_classes):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.savefig(stats_dir + 'Confusion_matrix_epoch_' + str(epoch_no))
    sys.stderr.write("Confusion matrix saved\n")


def test(data_file, batch_size, gpu_mode, trained_model, num_classes, num_workers, train_epoch, test_loss_file):
    """
    Test a trained model
    :param data_file: Test CSV file containing the test set
    :param batch_size: Batch size for prediction
    :param gpu_mode: If True GPU will be used
    :param trained_model: Trained model
    :param num_classes: Number of output classes (3- HOM, HET, HOM_ALT)
    :param num_workers: Number of workers for data loader
    :param train_epoch: Which train epoch has finished running
    :return:
    """
    transformations = transforms.Compose([transforms.ToTensor()])

    # data loader
    test_data = SequenceDataset(data_file, transformations)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=gpu_mode)
    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    # set the evaluation mode of the model
    test_model = trained_model.eval()
    if gpu_mode:
        test_model = test_model.cuda()

    # Loss
    test_criterion = nn.CrossEntropyLoss()

    # Test the Model
    sys.stderr.write(TextColor.PURPLE + 'Test starting\n' + TextColor.END)
    total_loss = 0
    total_images = 0
    batches_done = 0
    # confusion_matrix = torch.zeros(num_classes, num_classes)
    for i, (images, labels, position_progress) in enumerate(test_loader):
        if gpu_mode is True and images.size(0) % 8 != 0:
            continue

        images = Variable(images, volatile=True)
        labels = Variable(labels, volatile=True)
        if gpu_mode:
            images = images.cuda()
            labels = labels.cuda()

        total_sequences = images.size(2)
        for seq in range(total_sequences):
            start_seq_pos = seq
            end_seq_pos = FLANK_SIZE + seq + WINDOW_SIZE + FLANK_SIZE
            if end_seq_pos >= total_sequences:
                break

            # Forward + Backward + Optimize
            outputs = test_model(images[:, :, start_seq_pos:end_seq_pos, :])
            true_labels = labels[:, start_seq_pos + FLANK_SIZE:start_seq_pos + FLANK_SIZE + WINDOW_SIZE]

            # update the confusion matrix
            # batches = outputs.size(0)
            # seqs = outputs.size(1)
            # for batch in range(batches):
            #     for seq in range(seqs):
            #         preds = outputs[batch, seq, :].data
            #         true_label = labels[batch, seq].data.cpu().numpy()[0]
            #         top_n, top_i = preds.topk(1)
            #         predicted_label = top_i[0]
            #         confusion_matrix[true_label, predicted_label] += 1

            loss = test_criterion(outputs.contiguous().view(-1, num_classes), true_labels.contiguous().view(-1))

            # loss count
            total_loss += loss.data[0]
            total_images += (images.size(0))

        batches_done += 1
        if batches_done % 1 == 0:
            sys.stderr.write(TextColor.BLUE+'Batches done: ' + str(batches_done) + " / " + str(len(test_loader)) +
                             "\n" + TextColor.END)

    avg_loss = total_loss / total_images if total_images else 0
    test_loss_file.write(str(train_epoch) + "," + str(avg_loss) + "\n")

    # plot_confusion_matrix(num_classes, confusion_matrix, train_epoch)

    sys.stderr.write(TextColor.YELLOW+'Test Loss: ' + str(avg_loss) + "\n"+TextColor.END)


def train(train_file, validation_file, batch_size, epoch_limit, file_name, gpu_mode, num_workers, num_classes=6):
    """
    Train a model and save
    :param train_file: A CSV file containing train image information
    :param validation_file: A CSV file containing test image information
    :param batch_size: Batch size for training
    :param epoch_limit: Number of epochs to train on
    :param file_name: The model output file name
    :param gpu_mode: If true the model will be trained on GPU
    :param num_workers: Number of workers for data loading
    :param num_classes: Number of output classes
    :return:
    """
    train_loss_file = open(stats_dir + "train_loss.csv", 'w')
    test_loss_file = open(stats_dir + "test_loss.csv", 'w')
    train_loss_file.write("epoch,batch,loss\n")
    test_loss_file.write("epoch,loss\n")

    transformations = transforms.Compose([transforms.ToTensor()])

    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)
    train_data_set = SequenceDataset(train_file, transformations)
    train_loader = DataLoader(train_data_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=gpu_mode
                              )
    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)\

    model = SimpleModel(image_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    if gpu_mode:
        model = model.cuda()

    # Loss function
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0

    if gpu_mode:
        model = torch.nn.DataParallel(model).cuda()

    # Train the Model
    sys.stderr.write(TextColor.PURPLE + 'Training starting\n' + TextColor.END)
    for epoch in range(start_epoch, epoch_limit, 1):
        total_loss = 0
        total_images = 0
        epoch_start_time = time.time()
        start_time = time.time()
        batches_done = 0
        for i, (images, labels, positional_information) in enumerate(train_loader):
            if gpu_mode is True and images.size(0) % 8 != 0:
                continue

            images = Variable(images)
            labels = Variable(labels)
            if gpu_mode:
                images = images.cuda()
                labels = labels.cuda()

            x = images
            y = labels

            # WALK ON THE SEQUENCE
            total_sequences = x.size(2)
            for seq in range(total_sequences):
                start_seq_pos = seq
                end_seq_pos = FLANK_SIZE + seq + WINDOW_SIZE + FLANK_SIZE
                if end_seq_pos >= total_sequences:
                    break

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = model(x[:, :, start_seq_pos:end_seq_pos, :])
                true_labels = y[:, start_seq_pos+FLANK_SIZE:start_seq_pos+FLANK_SIZE+WINDOW_SIZE]

                loss = criterion(outputs.contiguous().view(-1, num_classes), true_labels.contiguous().view(-1))
                loss.backward()
                optimizer.step()

                # loss count
                total_loss += loss.data[0]
                total_images += (x.size(0))

            batches_done += 1

            if batches_done % 1 == 0:
                avg_loss = (total_loss / total_images) if total_images else 0
                sys.stderr.write(TextColor.BLUE + "EPOCH: " + str(epoch+1) + " Batches done: " + str(batches_done)
                                 + " / " + str(len(train_loader)) + "\n" + TextColor.END)
                sys.stderr.write(TextColor.YELLOW + "Loss: " + str(avg_loss) + "\n" + TextColor.END)
                sys.stderr.write(TextColor.DARKCYAN + "Time Elapsed: " + str(time.time() - start_time) +
                                 "\n" + TextColor.END)
                train_loss_file.write(str(epoch+1) + "," + str(batches_done) + "," + str(avg_loss) + "\n")
                start_time = time.time()

        avg_loss = (total_loss / total_images) if total_images else 0
        sys.stderr.write(TextColor.BLUE + "EPOCH: " + str(epoch+1) + " Completed\n" + TextColor.END)
        sys.stderr.write(TextColor.YELLOW + "Loss: " + str(avg_loss) + "\n" + TextColor.END)
        sys.stderr.write(TextColor.DARKCYAN + "Time Elapsed: " + str(time.time() - epoch_start_time)
                         + "\n" + TextColor.END)

        train_loss_file.write(str(epoch + 1) + "," + str(len(train_loader)) + "," + str(avg_loss) + "\n")

        # After each epoch do validation
        test(validation_file, batch_size, gpu_mode, model, num_classes, num_workers, epoch, test_loss_file)
        save_best_model(model, optimizer, file_name+"_epoch_"+str(epoch+1))

    sys.stderr.write(TextColor.PURPLE + 'Finished training\n' + TextColor.END)


def save_best_model(best_model, optimizer, file_name):
    """
    Save the best model
    :param best_model: A trained model
    :param optimizer: Optimizer
    :param file_name: Output file name
    :return:
    """
    sys.stderr.write(TextColor.BLUE + "SAVING MODEL.\n" + TextColor.END)
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
    parser.register("type", "bool", lambda v: v.lower() == "true")
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
    FLAGS, unparsed = parser.parse_known_args()
    model_dir, stats_dir = handle_output_directory(FLAGS.model_out.rpartition('/')[0]+"/")
    model_out_dir = model_dir + "seq2seq_model_"
    sys.stderr.write(TextColor.BLUE + "THE MODEL AND STATS LOCATION: " + str(model_dir) + "\n" + TextColor.END)
    train(FLAGS.train_file, FLAGS.test_file, FLAGS.batch_size, FLAGS.epoch_size, model_out_dir, FLAGS.gpu_mode,
          FLAGS.num_workers)
