import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
import pickle


class SequenceDataset(Dataset):
    """
    Arguments:
        A CSV file path
    """

    def __init__(self, csv_path, transform=None):
        data_frame = pd.read_csv(csv_path, header=None)
        assert data_frame[0].apply(lambda x: os.path.isfile(x.split(' ')[0]+'.png')).all(), \
            "Some images referenced in the CSV file were not found"
        self.transform = transform

        self.file_info = data_frame[0]
        self.position_info = data_frame[1]
        self.label = data_frame[2]
        self.reference_seq = data_frame[3]

    @staticmethod
    def load_dictionary(dictionary_location):
        f = open(dictionary_location, 'rb')
        dict = pickle.load(f)
        f.close()
        return dict

    def __getitem__(self, index):
        # load the image
        file_name, shape_y, shape_x, shape_z = self.file_info[index].split(' ')
        img = Image.open(file_name + '.png')
        np_array_of_img = np.array(img.getdata())
        img_shape = (int(shape_y), int(shape_x), int(shape_z))
        img = np.reshape(np_array_of_img, img_shape)

        # load positional information
        chromosome_name, genomic_start_position, slice_start, slice_end = self.position_info[index].split(' ')
        img = img[:, int(slice_start):int(slice_end), :]
        img = np.transpose(img, (1, 0, 2))

        # load the labels
        label = [ord(x)-ord('A') for x in self.label[index][20:40]]
        label = np.array(label)

        # load genomic position information
        reference_sequence = self.reference_seq[index][20:40]

        # type fix and convert to tensor
        img = img.astype(dtype=np.uint8)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(label)

        chromosome_name = chromosome_name

        positional_information = (chromosome_name, genomic_start_position, reference_sequence)

        return img, label, positional_information

    def __len__(self):
        return len(self.file_info.index)
