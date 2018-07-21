from __future__ import print_function
import argparse
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
import pickle
import sys
import os
import time

# Custom generator for our dataset
from modules.hyperband.hyperband import Hyperband
from modules.handlers.TextColor import TextColor
from modules.hyperband.train import train
from modules.hyperband.test import test
"""
Tune hyper-parameters of a model using hyperband.
Input:
- A train CSV file
- A test CSV file

Output:
- A model with tuned hyper-parameters
"""


class WrapHyperband:
    """
    Wrap hyperband around a model to tune hyper-parameters.
    """
    # Paramters of the model
    # depth=28 widen_factor=4 drop_rate=0.0
    def __init__(self, train_file, test_file, gpu_mode, model_out_dir, log_dir, max_epochs, batch_size, num_workers):
        """
        Initialize the object
        :param train_file: A train CSV file containing train data set information
        :param test_file: A test CSV file containing train data set information
        :param gpu_mode: If true, GPU will be used to train and test the models
        :param model_out_dir: Directory to save the model
        :param log_dir: Directory to save the log
        """
        # the hyper-parameter space is defined here
        self.space = {
            # hp.loguniform returns a value drawn according to exp(uniform(low, high)) so that the logarithm of the
            # return value is uniformly distributed.
            'encoder_lr': hp.loguniform('enc_lr', -12, -4),
            'decoder_lr': hp.loguniform('dec_lr', -12, -4),
            'encoder_l2': hp.loguniform('enc_l2', -12, -4),
            'decoder_l2': hp.loguniform('dec_l2', -12, -4),
            # 'hidden_size': hp.choice('hs', (128, 256, 512)),
            # 'gru_layers': hp.choice('gl', (1, 3, 5)),
        }
        self.train_file = train_file
        self.test_file = test_file
        self.gpu_mode = gpu_mode
        self.log_directory = log_dir
        self.model_out_dir = model_out_dir
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_params(self):
        """
        Get a random draw from the parameter space.
        :return: A randomly drawn sample space
        """
        return sample(self.space)

    def try_params(self, n_iterations, model_params):
        """
        Try a parameter space to train a model with n_iterations (epochs).
        :param n_iterations: Number of epochs to train on
        :param model_params: Parameter space
        :return: trained model, optimizer and stats dictionary (loss and others)
        """
        # Number of iterations or epoch for the model to train on
        n_iterations = int(round(n_iterations))
        params, retrain_model, retrain_model_path, prev_ite = model_params
        sys.stderr.write(TextColor.BLUE + '\nEpochs: ' + str(n_iterations) + "\n" + TextColor.END)
        sys.stderr.write(TextColor.BLUE + str(params) + "\n" + TextColor.END)

        num_workers = self.num_workers
        epoch_limit = int(n_iterations)
        hidden_size = 512
        gru_layers = 3
        batch_size = self.batch_size
        enc_lr = params['encoder_lr']
        enc_l2 = params['encoder_l2']
        dec_lr = params['decoder_lr']
        dec_l2 = params['decoder_l2']

        # train a model
        enc_model, dec_model, enc_optimizer, dec_optimizer, stats_dictionary = train(self.train_file, self.test_file,
                                                                                     batch_size, epoch_limit, prev_ite,
                                                                                     self.gpu_mode, num_workers,
                                                                                     retrain_model, retrain_model_path,
                                                                                     gru_layers, hidden_size,
                                                                                     enc_lr, enc_l2, dec_lr, dec_l2)
        # test the trained mode
        # stats_dictionary = test(self.test_file, batch_size, self.gpu_mode, enc_model, dec_model, num_workers,
        #                         hidden_size, num_classes=6)

        return enc_model, dec_model, enc_optimizer, dec_optimizer, stats_dictionary

    def run(self, save_output):
        """
        Run the hyper-parameter tuning algorithm
        :param save_output: If true, output will beb saved in a pkl file
        :return:
        """
        hyperband = Hyperband(self.get_params, self.try_params, max_iteration=self.max_epochs, downsample_rate=3,
                              model_directory=self.model_out_dir, log_directory=self.log_directory)
        results = hyperband.run()

        if save_output:
            with open('results.pkl', 'wb') as f:
                pickle.dump(results, f)

        # Print top 5 configs based on loss
        results = sorted(results, key=lambda r: r['loss'])[:5]
        print(results)


def handle_output_directory(output_dir):
    """
    Process the output directory and return a valid directory where we save the outputs
    :param output_dir: Output directory path
    :return:
    """
    # process the output directory
    if output_dir[-1] != "/":
        output_dir += "/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # create an internal directory so we don't overwrite previous runs
    timestr = time.strftime("%m%d%Y_%H%M%S")
    internal_directory = "hyperband_run_" + timestr + "/"
    output_dir = output_dir + internal_directory

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    model_output_dir = output_dir+'trained_models/'
    log_output_dir = output_dir+'logs/'
    os.mkdir(model_output_dir)
    os.mkdir(log_output_dir)

    return model_output_dir, log_output_dir


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
        "--gpu_mode",
        type=bool,
        default=False,
        help="If true then cuda is on."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default='./hyperband_output/',
        help="Directory to save the model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=100,
        help="Batch size for training, default is 100."
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        required=False,
        default=10,
        help="Epoch size for training iteration."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=40,
        help="Epoch size for training iteration."
    )
    FLAGS, unparsed = parser.parse_known_args()
    model_dir, log_dir = handle_output_directory(FLAGS.output_dir)
    wh = WrapHyperband(FLAGS.train_file, FLAGS.test_file, FLAGS.gpu_mode, model_dir, log_dir, FLAGS.max_epochs,
                       FLAGS.batch_size, FLAGS.num_workers)
    wh.run(save_output=True)
