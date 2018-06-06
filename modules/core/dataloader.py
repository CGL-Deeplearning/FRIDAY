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
        data_frame = pd.read_csv(csv_path, header=None, dtype=str)
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
        file_name, allele_dict_path, shape_y, shape_x, shape_z = self.file_info[index].split(' ')
        img = Image.open(file_name + '.png')
        np_array_of_img = np.array(img.getdata())
        img_shape = (int(shape_y), int(shape_x), int(shape_z))
        img = np.reshape(np_array_of_img, img_shape)
        img = np.transpose(img, (1, 0, 2))

        # load positional information
        chromosome_name, genomic_start_position = self.position_info[index].split(' ')

        # load the labels
        label = [int(x) for x in self.label[index]]
        label = np.array(label)

        # load genomic position information
        reference_sequence = self.reference_seq[index]

        # type fix and convert to tensor
        img = img.astype(dtype=np.uint8)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(label)

        chromosome_name = chromosome_name

        positional_information = (chromosome_name, genomic_start_position, reference_sequence, allele_dict_path)

        return img, label, positional_information

    def __len__(self):
        return len(self.file_info.index)
