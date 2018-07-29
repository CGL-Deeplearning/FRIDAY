import pandas as pd
import os
import sys
import h5py
from tqdm import tqdm

csv_path = sys.argv[1]

tmp_df = pd.read_csv(csv_path, header=None)
img_files = list(tmp_df[0])
for i in tqdm(range(len(img_files))):
    img_file = img_files[i]
    hdf5_file_path, allele_dict_path = img_file.split(' ')
    if os.path.isfile(img_file) is False:
        print("INVALID FILE PATH: ", img_file)
        exit()
    elif os.path.isfile(allele_dict_path) is False:
        print("INVALID FILE PATH: ", allele_dict_path)
        exit()
    else:
        hdf5_file = h5py.File(hdf5_file_path, 'r')

        if 'images' not in hdf5_file.keys():
            print("NO IMAGES IN HDF5", img_file)
            exit()
