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
        #     "Some images referenced in the CSV file were not found"
        self.transform = transform

        self.file_info = list(data_frame[0])
        self.index_info = list(data_frame[1])
        self.label = list(data_frame[3])

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
        img = np.array(image_dataset[hdf5_index], dtype=np.float32)
        hdf5_file.close()

        # load the labels
        label = self.label[index]
        label = [int(x) for x in label]
        label = np.array(label)

        # type fix and convert to tensor
        if self.transform is not None:
            img = self.transform(img)
            img = img.transpose(1, 2)

        label = torch.from_numpy(label)

        return img, label

    def __len__(self):
        return len(self.file_info)
