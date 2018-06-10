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
        assert data_frame[0].apply(lambda x: os.path.isfile(x.split(' ')[0])).all(), \
            "Some images referenced in the CSV file were not found"
        self.transform = transform

        self.file_info = data_frame[0]
        self.index_info = data_frame[1]
        self.position_info = data_frame[2]
        self.label = data_frame[3]
        self.reference_seq = data_frame[4]

    @staticmethod
    def load_dictionary(dictionary_location):
        f = open(dictionary_location, 'rb')
        dict = pickle.load(f)
        f.close()
        return dict

    def __getitem__(self, index):
        # load the image
        hdf5_file_path, allele_dict_path = self.file_info[index].split(' ')
        hdf5_index = int(self.index_info[index])
        hdf5_file = h5py.File(hdf5_file_path, 'r')
        image_dataset = hdf5_file['images']
        img = image_dataset[hdf5_index]
        # load positional information
        chromosome_name, genomic_start_position = self.position_info[index].split(' ')

        # load the labels
        label = [int(x) for x in self.label[index]]
        label = np.array(label)

        # load genomic position information
        reference_sequence = self.reference_seq[index]

        img = img.astype(dtype=np.uint8)
        # type fix and convert to tensor
        if self.transform is not None:
            img = self.transform(img)
            img = img.transpose(1, 2)

        label = torch.from_numpy(label)

        chromosome_name = chromosome_name

        positional_information = (chromosome_name, genomic_start_position, reference_sequence, allele_dict_path)

        return img, label, positional_information

    def __len__(self):
        return len(self.file_info.index)
