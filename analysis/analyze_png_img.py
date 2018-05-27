import os
import argparse
from PIL import Image, ImageOps
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
    if 117.0 <= support_color <= 131.0:
        return 2
    if 0.0 <= support_color <= 10.0:
        return 0


def get_alt_type_by_color(alt_type_color):
    if alt_type_color == 2.0:
        return 0
    if alt_type_color == 150.0:
        return 1
    if alt_type_color == 254.0:
        return 2
    if alt_type_color == 50.0:
        return 3


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
    if is_match == 50:
        return 1
    elif is_match == 254:
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


def analyze_array(img):
    img_h, img_w, img_c = img.shape
    print("BASE CHANNEL")
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
    print("BASE CHANNEL")
    for i in range(img_h):
        for j in range(img_w):
            if img[i][j][0] != 0:
                print(get_base_by_color(img[i][j][0]), end='')
            else:
                print(' ', end='')
        print()

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
    img = img * 255.0
    img_c, img_w, img_h = img.shape
    print("BASE CHANNEL")
    for i in range(img_h):
        for j in range(img_w):
            if img[0][j][i] != 0:
                print(get_base_by_color(img[0][j][i]), end='')
            else:
                print(' ', end='')
        print()

    print("SUPPORT CHANNEL")
    for i in range(img_h):
        for j in range(img_w):
            if img[6][j][i] != 0:
                print(get_alt_support_by_color(img[6][j][i]), end='')
            else:
                print(' ', end='')
        print()


if __name__ == '__main__':
    """
    Processes arguments and performs tasks to generate the pileup.
    """

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--img",
        type=str,
        required=True,
        help="Path to the image."
    )
    FLAGS, not_parsed_flags = parser.parse_known_args()
    # make output directory if not already created
    analyze_it(FLAGS.img)