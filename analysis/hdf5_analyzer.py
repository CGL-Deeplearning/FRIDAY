import argparse
import h5py


def get_read_base_from_color(base_color):
    # {'A': 25.0, 'C': 75.0, 'G': 125.0, 'T': 175.0, '*': 225.0, '-': 250.0, 'N': 10.0}
    if base_color == 25.0:
        return 'A'
    if base_color == 75.0:
        return 'C'
    if base_color == 125.0:
        return 'G'
    if base_color == 175.0:
        return 'T'
    if base_color == 225.0:
        return '*'
    if base_color == 250.0:
        return '-'
    if base_color == 10.0:
        return 'N'
    return ' '


def get_alt_support_by_color(support_color):
    """
    :return:
    """
    if 240.0 <= support_color <= 255.0:
        return 1
    if 125.0 <= support_color <= 135.0:
        return 2
    if 4 <= support_color <= 10:
        return 0
    return ' '


def get_allele_length(support_color):
    """
    :return:
    """
    if 240.0 <= support_color <= 255.0:
        return 2
    if 145.0 <= support_color <= 155.0:
        return 1
    if 4 <= support_color <= 10:
        return 0
    return ' '


def analyze_array(img):
    img_h, img_w, img_c = img.shape
    print("SUPPORT TYPE CHANNEL")
    for i in range(img_h):
        for j in range(img_w):
            print(get_read_base_from_color(img[i][j][0]), end='')
        print()

    print("ALLELE LENGTH")
    for i in range(img_h):
        for j in range(img_w):
            print(get_allele_length(img[i][j][8]), end='')
        print()

    print("ALLELE SUPPORT CHANNEL")
    for i in range(img_h):
        for j in range(img_w):
            print(get_alt_support_by_color(img[i][j][9]), end='')
        print()


def analyze_image(hdf5_file_path, index):
    hdf5_file = h5py.File(hdf5_file_path, 'r')
    image_dataset = hdf5_file['images']
    image = image_dataset[index]
    analyze_array(image)


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks to generate the pileup.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdf5_file",
        type=str,
        required=True,
        help="Bed file containing confident windows."
    )
    parser.add_argument(
        "--index",
        type=int,
        required=False,
        default=0,
        help="Bed file containing confident windows."
    )
    FLAGS, unparsed = parser.parse_known_args()
    analyze_image(FLAGS.hdf5_file, FLAGS.index)
