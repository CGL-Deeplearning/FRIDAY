import argparse
from PIL import Image
import numpy as np
import math


def get_base_by_color(base):
    """
    Get color based on a base.
    - Uses different band of the same channel.
    :param base:
    :return:
    """
    if 250.0 <= base <= 255.0:
        return 'A'
    if 99.0 <= base <= 105.0:
        return 'C'
    if 180.0 <= base <= 185.0:
        return 'G'
    if 25.0 <= base <= 35.0:
        return 'T'
    if 55.0 <= base <= 65.0:
        return '*'
    if 145.0 <= base <= 155.0:
        return '.'
    if 4.0 <= base <= 6.0:
        return 'N'
    if 0.0 <= base <= 3.0:
        return ' '


def get_alt_support_by_color(support_color):
    """
    :return:
    """
    if 240.0 <= support_color <= 255.0:
        return 1
    if 0.0 <= support_color <= 10.0:
        return 0


def get_alt_type_by_color(alt_type_color):
    if 1.0 <= alt_type_color <= 2.0:
        return '-'
    if 145.0 <= alt_type_color <= 155.0:
        return 'S'
    if 245.0 <= alt_type_color <= 260.0:
        return 'I'
    if 45.0 <= alt_type_color == 55.0:
        return 'D'


def get_quality_by_color(map_quality):
    """
    Get a color spectrum given mapping quality
    :param map_quality: value of mapping quality
    :return:
    """
    color = math.floor(((map_quality / 254) * 9))
    return color


def get_match_ref_color(is_match):
    """
    Get color for base matching to reference
    :param is_match: If true, base matches to reference
    :return:
    """
    if 45.0 <= is_match <= 55.0:
        return 1
    elif 250.0 <= is_match <= 255.0:
        return 0


def get_strand_color(is_rev):
    """
    Get color for forward and reverse reads
    :param is_rev: True if read is reversed
    :return:
    """
    if is_rev == 240.0:
        return 1
    else:
        return 0


def get_cigar_by_color(cigar_code):
    """
    ***NOT USED YET***
    :param is_in_support:
    :return:
    """
    if cigar_code == 254:
        return 0
    if cigar_code == 152:
        return 1
    if cigar_code == 76:
        return 2


def visualize_match_channel(img):
    img_h, img_w, img_c = img.shape
    img_h = 28
    entire_image = []
    for i in range(img_h):
        image_row = []
        for j in range(img_w):
            if img[i][j][4] != 0:
                print(get_match_ref_color(img[i][j][4]), end='')
                if get_match_ref_color(img[i][j][4]) == 0:
                    image_row.append([0, 255, 0, 255])
                if get_match_ref_color(img[i][j][4]) == 1:
                    image_row.append([0, 54, 0, 255])
            else:
                print(' ', end='')
                image_row.append([0, 0, 0, 255])
        entire_image.append(image_row)
        print()
    entire_image = np.array(entire_image)
    from scipy import misc
    misc.imsave("Mismatch_visualized" + ".png", entire_image, format="PNG")


def visualize_allele_channel(img):
    img_h, img_w, img_c = img.shape
    img_h = 28
    entire_image = []
    for i in range(img_h):
        image_row = []
        for j in range(img_w):
            if img[i][j][6] != 0:
                print(get_alt_support_by_color(img[i][j][6]), end='')
                if get_alt_support_by_color(img[i][j][6]) == 0:
                    image_row.append([54, 0, 0, 255])
                if get_alt_support_by_color(img[i][j][6]) == 1:
                    image_row.append([255, 0, 0, 255])
            else:
                print(' ', end='')
                image_row.append([0, 0, 0, 255])
        entire_image.append(image_row)
        print()
    entire_image = np.array(entire_image)
    from scipy import misc
    misc.imsave("Alt1_visualized" + ".png", entire_image, format="PNG")

    entire_image = []
    for i in range(img_h):
        image_row = []
        for j in range(img_w):
            if img[i][j][8] != 0:
                print(get_alt_support_by_color(img[i][j][8]), end='')
                if get_alt_support_by_color(img[i][j][8]) == 0:
                    image_row.append([0, 0, 54, 255])
                if get_alt_support_by_color(img[i][j][8]) == 1:
                    image_row.append([0, 0, 255, 255])
            else:
                print(' ', end='')
                image_row.append([0, 0, 0, 255])
        entire_image.append(image_row)
        print()
    entire_image = np.array(entire_image)
    from scipy import misc
    misc.imsave("Alt2_rgb" + ".png", entire_image, format="PNG")


def visualize_allele_rgb(img):
    img_h, img_w, img_c = img.shape
    img_h = 28
    entire_image = []
    for i in range(img_h):
        image_row = []
        for j in range(img_w):
            if img[i][j][6] != 0:
                print(get_alt_support_by_color(img[i][j][6]), end='')
                if get_alt_support_by_color(img[i][j][6]) == 0:
                    image_row.append([255, 255, 255, 255])
                if get_alt_support_by_color(img[i][j][6]) == 1:
                    if get_base_by_color(img[i][j][0]) == get_base_by_color(img[0][j][0]) and i > 0:
                        image_row.append([255, 255, 255, 255])
                    elif get_base_by_color(img[i][j][0]) == 'A':
                        image_row.append([0, 0, 255, 255])
                    elif get_base_by_color(img[i][j][0]) == 'C':
                        image_row.append([255, 0, 0, 255])
                    elif get_base_by_color(img[i][j][0]) == 'G':
                        image_row.append([0, 255, 0, 255])
                    elif get_base_by_color(img[i][j][0]) == 'T':
                        image_row.append([255, 255, 0, 255])
                    else:
                        # purple
                        image_row.append([160, 32, 240, 255])
            else:
                print(' ', end='')
                image_row.append([0, 0, 0, 255])
        entire_image.append(image_row)
        print()
    entire_image = np.array(entire_image)
    from scipy import misc
    misc.imsave("Alt1_rgb" + ".png", entire_image, format="PNG")

    entire_image = []
    for i in range(img_h):
        image_row = []
        for j in range(img_w):
            if img[i][j][8] != 0:
                print(get_alt_support_by_color(img[i][j][8]), end='')
                if get_alt_support_by_color(img[i][j][8]) == 0:
                    image_row.append([255, 255, 255, 255])
                if get_alt_support_by_color(img[i][j][8]) == 1:
                    if get_base_by_color(img[i][j][0]) == get_base_by_color(img[0][j][0]) and i > 0:
                        image_row.append([255, 255, 255, 255])
                    elif get_base_by_color(img[i][j][0]) == 'A':
                        image_row.append([0, 0, 255, 255])
                    elif get_base_by_color(img[i][j][0]) == 'C':
                        image_row.append([255, 0, 0, 255])
                    elif get_base_by_color(img[i][j][0]) == 'G':
                        image_row.append([0, 255, 0, 255])
                    elif get_base_by_color(img[i][j][0]) == 'T':
                        image_row.append([255, 255, 0, 255])
                    else:
                        # purple
                        image_row.append([160, 32, 240, 255])
            else:
                print(' ', end='')
                image_row.append([0, 0, 0, 255])
        entire_image.append(image_row)
        print()
    entire_image = np.array(entire_image)
    from scipy import misc
    misc.imsave("Alt2_rgb" + ".png", entire_image, format="PNG")


def visualize_map_quality_channel(img):
    img_h, img_w, img_c = img.shape
    img_h = 28
    entire_image = []
    for i in range(img_h):
        image_row = []
        for j in range(img_w):
            image_row.append([0, 0, img[i][j][1], 255])
        entire_image.append(image_row)
    entire_image = np.array(entire_image)
    from scipy import misc
    misc.imsave("Map_quality_visualized" + ".png", entire_image, format="PNG")


def visualize_base_channel(img):
    img_h, img_w, img_c = img.shape
    img_h = 28
    entire_image = []
    for i in range(img_h):
        image_row = []
        for j in range(img_w):
            if img[i][j][0] != 0:
                print(get_base_by_color(img[i][j][0]), end='')
                if get_base_by_color(img[i][j][0]) == get_base_by_color(img[0][j][0]) and i > 0:
                    image_row.append([255, 255, 255, 255])
                elif get_base_by_color(img[i][j][0]) == 'A':
                    image_row.append([0, 0, 255, 255])
                elif get_base_by_color(img[i][j][0]) == 'C':
                    image_row.append([255, 0, 0, 255])
                elif get_base_by_color(img[i][j][0]) == 'G':
                    image_row.append([0, 255, 0, 255])
                elif get_base_by_color(img[i][j][0]) == 'T':
                    image_row.append([255, 255, 0, 255])
                else:
                    # purple
                    image_row.append([160, 32, 240, 255])
            else:
                print(' ', end='')
                image_row.append([0, 0, 0, 255])
        entire_image.append(image_row)
        print()
    entire_image = np.array(entire_image)
    from scipy import misc
    misc.imsave("Base_visualized" + ".png", entire_image, format="PNG")


def analyze_array(img):
    # visualize_base_channel(img)
    # visualize_match_channel(img)
    # visualize_map_quality_channel(img)
    # visualize_allele_channel(img)
    # visualize_allele_rgb(img)
    # return

    img_h, img_w, img_c = img.shape
    for i in range(img_h):
        for j in range(img_w):
            if img[i][j][0] != 0:
                print(get_base_by_color(img[i][j][0]), end='')
            else:
                print(' ', end='')
        print()

    print("SUPPORT CHANNEL")
    for i in range(img_h):
        for j in range(img_w):
            if img[i][j][6] != 0:
                print(get_alt_support_by_color(img[i][j][6]), end='')
            else:
                print(' ', end='')
        print()
    print("SUPPORT TYPE CHANNEL")
    for i in range(img_h):
        for j in range(img_w):
            if img[i][j][7] != 0:
                print(get_alt_type_by_color(img[i][j][7]), end='')
            else:
                print(' ', end='')
        print()


def analyze_it(img, shape):
    file = img
    img = Image.open(file)
    np_array_of_img = np.array(img.getdata())

    img = np.reshape(np_array_of_img, shape)
    img = np.transpose(img, (0, 1, 2))
    img_h, img_w, img_c = img.shape
    # print("BASE CHANNEL")
    for i in range(img_h):
        for j in range(img_w):
            if img[i][j][0] != 0:
                print(get_base_by_color(img[i][j][0]), end='')
            else:
                print(' ', end='')
        print()
    return
    print("ALLELE SUPPORT CHANNEL")
    for i in range(img_h):
        for j in range(img_w):
            if img[i][j][6] != 0:
                print(get_alt_support_by_color(img[i][j][6]), end='')
            else:
                print(' ', end='')
        print()

    print("SUPPORT TYPE CHANNEL")
    for i in range(img_h):
        for j in range(img_w):
            if img[i][j][7] != 0:
                print(get_alt_type_by_color(img[i][j][7]), end='')
            else:
                print(' ', end='')
        print()


def analyze_tensor(img):
    img_c, img_w, img_h = img.shape
    print("BASE CHANNEL")
    for i in range(img_h):
        for j in range(img_w):
            if img[0][j][i] != 0:
                # print(img[0][j][i])
                print(get_base_by_color(img[0][j][i]), end='')
            else:
                print(' ', end='')
                # print(' ')
        print()

    print("SUPPORT CHANNEL")
    for i in range(img_h):
        for j in range(img_w):
            if img[6][j][i] != 0:
                print(get_alt_support_by_color(img[6][j][i]), end='')
            else:
                print(' ', end='')
        print()


def analyze_hdf5(img):
    print("BASE CHANNEL")
    for i in range(img_h):
        for j in range(img_w):
            if img[0][j][i] != 0:
                # print(img[0][j][i])
                print(get_base_by_color(img[0][j][i]), end='')
            else:
                print(' ', end='')
                # print(' ')
        print()

    print("SUPPORT CHANNEL")
    for i in range(img_h):
        for j in range(img_w):
            if img[6][j][i] != 0:
                print(get_alt_support_by_color(img[6][j][i]), end='')
            else:
                print(' ', end='')
        print()


def get_ref_base_from_color(base_color):
    # 'A': 25.0, 'C': 75.0, 'G': 125.0, 'T': 175.0, '*': 225.0
    if base_color == 0.0:
        return '.'
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


def get_read_base_from_color(base_color):
    # 'A': 25.0, 'C': 75.0, 'G': 125.0, 'T': 175.0, '*': 225.0
    if 25.0 <= base_color < 75.0:
        base_color -= 25.0
    elif 75.0 <= base_color < 125.0:
        base_color -= 75.0
    elif 125.0 <= base_color < 175.0:
        base_color -= 125.0
    elif 175.0 <= base_color < 225.0:
        base_color -= 175.0
    elif 225.0 <= base_color < 255.0:
        base_color -= 225.0

    # {'A': 0.0, 'C': 5.0, 'G': 10.0, 'T': 15.0, '*': 20.0, '-': 25.0}
    if base_color == 0.0:
        return 'A'
    if base_color == 5.0:
        return 'C'
    if base_color == 10.0:
        return 'G'
    if base_color == 15.0:
        return 'T'
    if base_color == 20.0:
        return '*'
    if base_color == 25.0:
        return '-'


def analyze_v3_images(img):
    img_h, img_w, img_c = img.shape
    img_h = 50
    entire_image = []
    for i in range(img_h):
        image_row = []
        for j in range(img_w):
            if i < 5:
                print(get_ref_base_from_color(img[i][j][0]), end='')
            elif img[i][j][0] == 0:
                print('.', end='')
            else:
                print(get_read_base_from_color(img[i][j][0]), end='')

            image_row.append([img[i][j][0], img[i][j][8], 0, 255])
        print()
        entire_image.append(image_row)
    entire_image = np.array(entire_image)
    from scipy import misc
    misc.imsave("Base_visualized_v3" + ".png", entire_image, format="PNG")
# import h5py
# hdf5_file = h5py.File("/data/users/kishwar/train_data/image_output/run_08102018_161950/19/19_259396.h5", 'r')
# image_dataset = hdf5_file['images']
# img = np.array(image_dataset[0], dtype=np.int32)
# from torchvision import transforms
# transformations = transforms.Compose([transforms.ToTensor()])
# img = transformations(img)
# img = img.transpose(1, 2)
# analyze_tensor(img)
