import argparse
import os
import time

# Custom generator for our dataset
from modules.models.train import train
"""
Input:
- A train CSV file
- A test CSV file

Output:
- A model with tuned hyper-parameters
"""


class TrainModule:
    """
    Train module
    """
    def __init__(self, train_file, test_file, gpu_mode, max_epochs, batch_size, num_workers,
                 retrain_model, retrain_model_path, model_dir, stats_dir):
        self.train_file = train_file
        self.test_file = test_file
        self.gpu_mode = gpu_mode
        self.log_directory = log_dir
        self.model_dir = model_dir
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.retrain_model = retrain_model
        self.retrain_model_path = retrain_model_path
        self.stats_dir = stats_dir
        # {'decoder_lr': 7.020638355820418e-05, 'decoder_l2': 8.806341254262352e-05,
        # 'encoder_lr': 0.0003090327459310237,  'encoder_l2': 0.0006757277458529228}
        # window =1
        # Params: {'decoder_l2': 0.0002892824841962342, 'encoder_l2': 0.00010703410259778772,
        #          'encoder_lr': 0.0003001654397610769, 'decoder_lr': 0.00013044660278148493}
        self.hidden_size = 512
        self.gru_layers = 3
        self.encoder_lr = 0.0003001654397610769
        self.encoder_l2 = 0.00010703410259778772
        self.decoder_lr = 0.00013044660278148493
        self.decoder_l2 = 0.0002892824841962342

    def train_model(self):
        # train a model
        enc_model, dec_model, enc_optimizer, dec_optimizer, stats_dictionary = train(self.train_file,
                                                                                     self.test_file,
                                                                                     self.batch_size,
                                                                                     self.epochs,
                                                                                     self.gpu_mode,
                                                                                     self.num_workers,
                                                                                     self.retrain_model,
                                                                                     self.retrain_model_path,
                                                                                     self.gru_layers,
                                                                                     self.hidden_size,
                                                                                     self.encoder_lr,
                                                                                     self.encoder_l2,
                                                                                     self.decoder_lr,
                                                                                     self.decoder_l2,
                                                                                     self.model_dir,
                                                                                     self.stats_dir,
                                                                                     train_mode=True)

        return enc_model, dec_model, enc_optimizer, dec_optimizer, stats_dictionary


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
        "--retrain_model_path",
        type=str,
        default=False,
        help="Path to the model that will be retrained."
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
    model_out_dir, log_dir = handle_output_directory(FLAGS.model_out)
    tm = TrainModule(FLAGS.train_file, FLAGS.test_file, FLAGS.gpu_mode, FLAGS.epoch_size, FLAGS.batch_size,
                     FLAGS.num_workers, FLAGS.retrain_model, FLAGS.retrain_model_path, model_out_dir, log_dir)
    tm.train_model()
