import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import h5py
import torch
import pickle


class SequenceDataset(Dataset):
    """
    Arguments:
        A CSV file path
    """

    def __init__(self, csv_path, transform=None):
        data_frame = pd.read_csv(csv_path, header=None, dtype=str)
        # assert data_frame[0].apply(lambda x: os.path.isfile(x.split(' ')[0])).all(), \
        # "Some images referenced in the CSV file were not found"
        self.transform = transform

        self.file_info = list(data_frame[0])
        self.index_info = list(data_frame[1])
        self.position_info = list(data_frame[2])
        self.label = list(data_frame[3])
        self.reference_seq = list(data_frame[4])

    @staticmethod
    def load_dictionary(dictionary_location):
        f = open(dictionary_location, 'rb')
        dict = pickle.load(f)
        f.close()
        return dict

    def __getitem__(self, index):
        # load the image
        hdf5_file_path, allele_dict_path = self.file_info[index].split(' ')
        # load positional information
        chromosome_name, genomic_start_position = self.position_info[index].split(' ')
        # load genomic position information
        reference_sequence = self.reference_seq[index]
        # load the labels
        label = self.label[index]
        label = [int(x) for x in label]

        label = np.array(label)

        label = torch.from_numpy(label)

        positional_information = (chromosome_name, genomic_start_position, reference_sequence, allele_dict_path)

        return label, positional_information

    def __len__(self):
        return len(self.file_info)