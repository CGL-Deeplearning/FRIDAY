from collections import defaultdict
from modules.handlers.ImageChannels import ImageChannels
from scipy import misc
import sys
import random
import collections
import numpy as np
from modules.handlers.TextColor import TextColor
import h5py
import operator
"""
Generate image and label of that image given a region. 
"""

# Debugging configuration
DEFAULT_MIN_MAP_QUALITY = 1
ALLELE_DEBUG = False
ALLELE_FREQUENCY_THRESHOLD_FOR_REPORTING = 0.5

# Data processing configuration
MIN_DELETE_QUALITY = 20
VCF_INDEX_BUFFER = -1

# Per sequence threshold
# jump window size so the last 50 bases will be overlapping
WINDOW_OVERLAP_JUMP = 10
# image size
WINDOW_SIZE = 10
# flanking size is the amount add on each size
CONTEXT_SIZE = 10
# boundary columns is the number of bases we process for safety
BOUNDARY_COLUMNS = 50
# ALL_HOM_BASE_RATIO = 0.005 (this worked great)
ALL_HOM_BASE_RATIO = 1
# buffer around boundary to make sure all the bases in the interval is included
POS_BUFFER = 0

# Logging configuration
LOG_LEVEL_HIGH = 1
LOG_LEVEL_LOW = 0
LOG_LEVEL = LOG_LEVEL_LOW
WARN_COLOR = TextColor.RED

PLOIDY = 2
SNP = 1
IN = 2
DEL = 3

HOM = 0
HET = 1
HOM_ALT = 2


class ImageGenerator:
    """
    Generate images of an interval
    """
    def __init__(self, candidate_finder_object):
        """
        Initialize dictionaries and object files
        :param candidate_finder_object: Candidate finder object that contains populated dictionaries for that region
        :param vcf_file: Path to the VCF file
        """
        self.pos_dicts = candidate_finder_object
        self.chromosome_name = candidate_finder_object.chromosome_name

        """
        Convert all positional dictionaries to indexed dictionaries.
        Positions: Genomic positions. For inserts they are same as the anchors
        Indexes: Index of a column, as we see inserts the positions increase.
        For example:
        ACT***AAA
        The genomic positions would be: {A: 1, C: 2, T: 3, *: 3, *: 3, *: 3, A: 4, A: 5, A: 6}
        The indices would be: {A: 1, C: 2, T: 3, *: 4, *: 5, *: 6, A: 7, A: 8, A: 9}
        """
        # for image generation
        self.top_alleles = defaultdict(list)
        self.image_row_for_reads = defaultdict(tuple)
        self.image_row_for_ref = defaultdict(tuple)
        self.positional_info_index_to_position = defaultdict(tuple)
        self.positional_info_position_to_index = defaultdict(int)
        self.base_frequency = defaultdict(lambda: defaultdict(int))
        self.index_based_coverage = defaultdict(int)
        self.reference_base_by_index = defaultdict(int)
        self.vcf_positional_dict = defaultdict(int)
        self.reference_string = ''

    def get_support_for_read(self, read_id, read_start_pos, read_end_pos):
        support_dict = defaultdict(tuple)
        pos = read_start_pos
        while pos < read_end_pos:
            candidate_alleles = self.pos_dicts.positional_allele_frequency[pos] \
                if pos in self.pos_dicts.positional_allele_frequency else None
            if candidate_alleles is not None:
                if pos not in self.top_alleles:
                    self.top_alleles[pos] = \
                        sorted(self.pos_dicts.positional_allele_frequency[pos].items(), key=operator.itemgetter(1, 0),
                               reverse=True)[:PLOIDY]
                support_candidate_type = SNP
                supported_allele = ''
                for counter, allele_info in enumerate(self.top_alleles[pos]):
                    allele, freq = allele_info
                    alt_allele, allele_type = allele
                    read_allele = ''
                    if allele_type == SNP:
                        # if there is a base in that position for that read
                        if pos in self.pos_dicts.base_dictionary[read_id]:
                            # get the base and the base quality
                            read_base, base_q = self.pos_dicts.base_dictionary[read_id][pos]
                            read_allele = read_base
                    elif allele_type == IN:
                        if pos in self.pos_dicts.base_dictionary[read_id]:
                            # get the base and the base quality
                            base, base_q = self.pos_dicts.base_dictionary[read_id][pos]
                            read_allele = read_allele + base
                        # if this specific read has an insert
                        if read_id in self.pos_dicts.insert_dictionary and \
                                pos in self.pos_dicts.insert_dictionary[read_id]:
                            # insert bases and qualities
                            in_bases, in_qualities = self.pos_dicts.insert_dictionary[read_id][pos]
                            read_allele = read_allele + in_bases
                    elif allele_type == DEL:
                        del_len = len(alt_allele)
                        alt_allele = alt_allele[0] + '*' * (del_len - 1)
                        i = pos
                        while i in self.pos_dicts.base_dictionary[read_id]:
                            base, base_q = self.pos_dicts.base_dictionary[read_id][i]
                            if i > pos and base != '*':
                                break
                            read_allele = read_allele + base
                            i += 1

                    if read_allele == alt_allele:
                        support_candidate_type = allele_type
                        supported_allele = alt_allele
                        support_dict[pos] = (counter+1, allele_type, alt_allele)
                        break

                if support_candidate_type == DEL:
                    pos += len(supported_allele) - 1
                else:
                    pos += 1
            else:
                pos += 1

        return support_dict

    def post_process_reads(self, read_id_list, interval_start, interval_end):
        """
        After all the inserts are processed, process the reads again to make sure all the in-del lengths match.
        :param read_id_list: List of read ids
        :return:
        """
        for read_id in read_id_list:
            start_pos, end_pos, mapping_quality, strand_direction = self.pos_dicts.read_info[read_id]
            start_pos_new = max(start_pos, interval_start)
            end_pos_new = min(end_pos, interval_end)
            read_to_image_row = []
            support_dict = self.get_support_for_read(read_id, start_pos, end_pos)

            for pos in range(start_pos_new, end_pos_new):
                if pos < interval_start:
                    continue

                if pos > interval_end:
                    break

                if pos not in self.pos_dicts.base_dictionary[read_id] and \
                        pos not in self.pos_dicts.insert_dictionary[read_id]:
                    print(pos, read_id)
                    continue

                if pos in support_dict:
                    support_allele_no, support_allele_type, support_allele = support_dict[pos]
                    # print(pos, support_allele_type, support_allele, support_allele_no)
                else:
                    support_allele_type = 0
                    support_allele_no = 0

                # if there is a base in that position for that read
                if pos in self.pos_dicts.base_dictionary[read_id]:
                    # get the base and the base quality
                    base, base_q = self.pos_dicts.base_dictionary[read_id][pos]
                    # see if the base is a delete
                    cigar_code = 0 if base != '*' else 1
                    # get the reference base of that position
                    ref_base = self.pos_dicts.reference_dictionary[pos]
                    # combine all the pileup attributes we want to encode in the image
                    pileup_attributes = (base, base_q, mapping_quality, cigar_code, strand_direction, support_allele_no,
                                         support_allele_type)
                    # create a channel object to covert these features to a pixel
                    channel_object = ImageChannels(pileup_attributes, ref_base)
                    # add the pixel to the row
                    read_to_image_row.append(channel_object.get_channels())
                    index_of_position = self.positional_info_position_to_index[pos]
                    # increase the coverage
                    self.index_based_coverage[index_of_position] += 1
                    if base == '*':
                        self.base_frequency[index_of_position]['.'] += 1
                    else:
                        self.base_frequency[index_of_position][base] += 1

                # if there's an insert
                if pos in self.pos_dicts.insert_length_info:
                    # get the length of insert
                    length_of_insert = self.pos_dicts.insert_length_info[pos]
                    total_insert_bases = 0
                    # if this specific read has an insert
                    if read_id in self.pos_dicts.insert_dictionary and pos in self.pos_dicts.insert_dictionary[read_id]:
                        # insert bases and qualities
                        in_bases, in_qualities = self.pos_dicts.insert_dictionary[read_id][pos]
                        total_insert_bases = len(in_bases)
                        # iterate through each of the bases and add those to the image
                        for i in range(total_insert_bases):
                            base = in_bases[i]
                            base_q = in_qualities[i]
                            cigar_code = 2
                            ref_base = ''
                            pileup_attributes = (base, base_q, mapping_quality, cigar_code, strand_direction, 0, 0)
                            channel_object = ImageChannels(pileup_attributes, ref_base)
                            read_to_image_row.append(channel_object.get_channels())

                            self.base_frequency[self.positional_info_position_to_index[pos] + i + 1][base] += 1
                            self.index_based_coverage[self.positional_info_position_to_index[pos] + i + 1] += 1

                    # if there's any other read that has a longer insert then you need to append
                    if length_of_insert > total_insert_bases:
                        # count the total number of bases you need to append
                        dot_bases = length_of_insert - total_insert_bases
                        # append those bases
                        for i in range(dot_bases):
                            base = '*'
                            base_q = MIN_DELETE_QUALITY
                            cigar_code = 2
                            ref_base = ''
                            pileup_attributes = (base, base_q, mapping_quality, cigar_code, strand_direction, 0, 0)
                            channel_object = ImageChannels(pileup_attributes, ref_base)
                            read_to_image_row.append(channel_object.get_channels())

                            indx = self.positional_info_position_to_index[pos] + total_insert_bases + i + 1
                            self.base_frequency[indx][base] += 1
                            self.index_based_coverage[indx] += 1

            self.image_row_for_reads[read_id] = (read_to_image_row, start_pos_new, end_pos_new)

    def post_process_reference(self, interval_start, interval_end):
        """
        Post process the reference with inserts to create the reference row.

        This also populates the indices so this should be run first while generating images.
        :return:
        """
        # find the start and end position for the reference
        left_position = interval_start
        right_position = interval_end + 1

        reference_to_image_row = []
        index = 0
        reference_string = ''
        for pos in range(left_position, right_position):
            # get the reference base for that position
            base = self.pos_dicts.reference_dictionary[pos] if pos in self.pos_dicts.reference_dictionary else 'N'
            # get the pixel value fot the reference
            pixel_values = ImageChannels.get_channels_for_ref(base)
            reference_to_image_row.append(pixel_values)
            # save index values
            self.positional_info_index_to_position[index] = (pos, False)
            self.positional_info_position_to_index[pos] = index
            self.reference_base_by_index[index] = base
            reference_string += base
            index += 1
            # if there's an insert add those insert bases
            if pos in self.pos_dicts.insert_length_info:
                for i in range(self.pos_dicts.insert_length_info[pos]):
                    base = '*'
                    pixel_values = ImageChannels.get_channels_for_ref(base)
                    reference_to_image_row.append(pixel_values)
                    self.positional_info_index_to_position[index] = (pos, True)
                    self.reference_base_by_index[index] = base
                    reference_string += base
                    index += 1
        # print(reference_to_image_row.shape)
        self.image_row_for_ref = (reference_to_image_row, left_position, right_position)
        self.reference_string = reference_string

    def get_reference_row(self, start_pos, end_pos):
        """
        Get the reference row of pixels for the image
        :param start_pos: Start genomic position
        :param end_pos: End genomic position
        :return:
        """
        ref_row, ref_start, ref_end = self.image_row_for_ref
        # find start and end index for the genomic region
        st_index = self.positional_info_position_to_index[start_pos] - self.positional_info_position_to_index[ref_start]
        end_index = self.positional_info_position_to_index[end_pos] - self.positional_info_position_to_index[ref_start]

        ref_row = np.array(ref_row[st_index:end_index])

        return ref_row

    def get_read_row(self, read_id, read_info, image_start, image_end):
        """
        Get the read row to add to the image
        :param read_id: Unique read id
        :param read_info: Read information
        :param image_start: Start position of the image
        :param image_end: End position of the image
        :return:
        """
        read_start, read_end, mq, is_rev = read_info
        read_row = self.image_row_for_reads[read_id][0]

        read_start_new = read_start
        read_end_new = read_end
        if image_start > read_start:
            read_start_new = image_start

        if image_end < read_end:
            read_end_new = image_end

        start_index = self.positional_info_position_to_index[read_start_new] - \
                      self.positional_info_position_to_index[read_start]
        end_index = self.positional_info_position_to_index[read_end_new] - \
                      self.positional_info_position_to_index[read_start]
        # print(start_index, end_index)
        # print(image_start, image_end)
        # print(read_start, read_end)
        # print(read_start_new, read_end_new)
        # exit()
        image_row = read_row[start_index:end_index]

        if image_start < read_start_new:
            distance = self.positional_info_position_to_index[read_start_new] - \
                      self.positional_info_position_to_index[image_start]
            empty_channels_list = [ImageChannels.get_empty_channels()] * int(distance)
            image_row = empty_channels_list + image_row

        return image_row, read_start_new, read_end_new

    @staticmethod
    def get_row_for_read(read_start, read_end, row_info, image_height):
        """
        Heuristically get a row for packing the read
        :param read_start: Start position of the read
        :param read_end: End position of the read
        :param row_info: Information about rows of the image (current packing situation)
        :param image_height: Height of the image
        :return:
        """
        for i in range(image_height):
            # if the read starts to the left of where all previous reads ended on this row, then select that row
            if read_start > row_info[i]:
                return i

        return -1

    def create_image(self, interval_start, interval_end, read_id_list, image_height=100):
        """
        Create image of a given region
        :param interval_start: Start of the interval (genomic position)
        :param interval_end: End of the interval (genomic position)
        :param read_id_list: List of reads that are aligned to this interval
        :param image_height: Height of the image
        :return:
        """

        image_row_info = defaultdict(int)

        # get the reference row
        ref_row = self.get_reference_row(interval_start, interval_end)

        # the image width is the width of the reference
        image_width = ref_row.shape[0]
        image_channels = ref_row.shape[1]

        whole_image = np.zeros((image_height, image_width, image_channels))
        # add the reference row as the first row of the image [0th row]
        whole_image[0, :, :] = np.array(ref_row)

        # update the packing information
        image_row_info[0] = interval_end

        # go through each of the reads and add them to the image
        for read_id in read_id_list:
            read_info = self.pos_dicts.read_info[read_id]
            read_start, read_end, mq, is_rev = read_info

            # get the row of the read
            row = self.get_row_for_read(read_start, read_end, image_row_info, image_height)

            # if you can't fit the read then skip this read
            if row < 0:
                continue

            # find the image start
            image_start = max(image_row_info[row], interval_start)
            image_index_read_start = self.positional_info_position_to_index[image_start] - \
                                         self.positional_info_position_to_index[interval_start]
            # get the row with the images
            read_row, read_start, read_end = self.get_read_row(read_id, read_info, image_start, interval_end)

            image_index_read_end = image_index_read_start + len(read_row)

            if image_index_read_end - image_index_read_start <= 0:
                continue

            # append the read to the row we want to pack it to
            whole_image[row, image_index_read_start:image_index_read_end, :] = np.array(read_row, dtype=np.float)
            # update packing information
            image_row_info[row] = read_end

        return whole_image

    @staticmethod
    def save_image_as_png(pileup_array, save_dir, file_name):
        """
        Save image as png
        :param pileup_array: The image
        :param save_dir: Directory path
        :param file_name: Name of the file
        :return:
        """
        pileup_array_2d = pileup_array.reshape((pileup_array.shape[0], -1))
        try:
            misc.imsave(save_dir + file_name + ".png", pileup_array_2d, format="PNG")
        except:
            sys.stderr.write(TextColor.RED)
            sys.stderr.write("ERROR: ERROR SAVING FILE: " + file_name + ".png" + "\n" + TextColor.END)
            sys.stderr.write()

    def get_label_sequence(self, interval_start, interval_end):
        """
        Get the label of a genomic interval
        :param interval_start: Interval start
        :param interval_end: Interval end
        :return:
        """
        start_index = self.positional_info_position_to_index[interval_start]
        end_index = self.positional_info_position_to_index[interval_end]
        # label sequence
        reference_string = ''
        label_sequence = ''

        for i in range(start_index, end_index):
            # build the reference string
            reference_string += self.reference_string[i]
            # from the positional vcf to the label string
            label_sequence = label_sequence + str(self.vcf_positional_dict[i])

        return label_sequence, reference_string

    @staticmethod
    def get_genotype_from_vcf_tuple(vcf_tuple):
        if vcf_tuple[0] != vcf_tuple[1]:
            return HET
        if vcf_tuple[0] == vcf_tuple[1] and vcf_tuple[0] != 0:
            return HOM_ALT
        return HOM

    @staticmethod
    def get_site_label_from_allele_tuple(pos, allele_tuple):
        base_to_letter_map = {'0/0': 0, '0/1': 1, '1/1': 2, '0/2': 3, '2/2': 4, '1/2': 5}
        if allele_tuple[1] == HOM and allele_tuple[2] == HOM:
            return base_to_letter_map['0/0']
        elif allele_tuple[1] == HET and allele_tuple[2] == HET:
            return base_to_letter_map['1/2']
        elif allele_tuple[1] == HET:
            return base_to_letter_map['0/1']
        elif allele_tuple[1] == HOM_ALT:
            return base_to_letter_map['1/1']
        elif allele_tuple[2] == HET:
            return base_to_letter_map['0/2']
        elif allele_tuple[2] == HOM_ALT:
            return base_to_letter_map['2/2']
        elif allele_tuple[1] == HOM_ALT and allele_tuple[2] == HOM_ALT:
            sys.stderr.write("WARN: INVALID VCF RECORD FOUND " + str(pos) + " " + str(allele_tuple) + "\n")

    def populate_vcf_alleles(self, positional_vcf):
        """
        From positional VCF alleles, populate the positional dictionary.
        :param positional_vcf: Positional VCF dictionar
        :return:
        """
        for pos in positional_vcf.keys():
            # get bam position
            bam_pos = pos + VCF_INDEX_BUFFER

            # we we haven't processed the position, we can't assign alleles
            if bam_pos not in self.positional_info_position_to_index:
                continue
            indx = self.positional_info_position_to_index[bam_pos]

            alt_alleles_found = self.top_alleles[bam_pos] \
                if bam_pos in self.top_alleles else []

            vcf_alts = []

            snp_recs, in_recs, del_recs = positional_vcf[pos]
            # SNP records
            for snp_rec in snp_recs:
                vcf_alts.append((snp_rec.alt[0], SNP, snp_rec.genotype))

            # insert record
            for in_rec in in_recs:
                vcf_alts.append((in_rec.alt, IN, in_rec.genotype))

            # delete record
            for del_rec in del_recs:
                # for delete reference holds which bases are deleted hence that's the alt allele
                vcf_alts.append((del_rec.ref, DEL, del_rec.genotype))

            alts_with_genotype = {1: 0, 2: 0}
            for counter, allele in enumerate(alt_alleles_found):
                allele_tuple = (allele[0])
                for vcf_allele in vcf_alts:
                    vcf_tuple = (vcf_allele[0], vcf_allele[1])
                    if allele_tuple == vcf_tuple:
                        alts_with_genotype[counter+1] = self.get_genotype_from_vcf_tuple(vcf_allele[2])

            self.vcf_positional_dict[indx] = self.get_site_label_from_allele_tuple(pos, alts_with_genotype)

    def get_segmented_image_sequences(self, interval_start, interval_end, positional_variants, read_id_list,
                                      file_info):
        """
        Generates segmented image sequences for training
        :param interval_start: Genomic interval start
        :param interval_end: Genomic interval stop
        :param positional_variants: VCF positional variants
        :param read_id_list: List of reads ids that fall in this region
        :param file_info: File names of hdf5 file and allele dict to save in summary
        :return:
        """
        # post process reference and read and label
        self.post_process_reference(interval_start - BOUNDARY_COLUMNS, interval_end + BOUNDARY_COLUMNS)
        self.post_process_reads(read_id_list, interval_start - BOUNDARY_COLUMNS, interval_end + BOUNDARY_COLUMNS)
        self.populate_vcf_alleles(positional_variants)
        # get the image
        image = self.create_image(interval_start - BOUNDARY_COLUMNS, interval_end + BOUNDARY_COLUMNS, read_id_list)
        label_seq, ref_seq = self.get_label_sequence(interval_start - BOUNDARY_COLUMNS, interval_end + BOUNDARY_COLUMNS)

        summary_strings = ''
        sliced_images = []
        ref_row, ref_start, ref_end = self.image_row_for_ref
        img_started_in_indx = self.positional_info_position_to_index[interval_start - BOUNDARY_COLUMNS] - \
                              self.positional_info_position_to_index[ref_start]

        img_ended_in_indx = self.positional_info_position_to_index[interval_end + BOUNDARY_COLUMNS] - \
                            self.positional_info_position_to_index[ref_start]

        # this is sliding window based approach
        image_index = 0
        img_w, img_h, img_c = 0, 0, 0

        for i, pos in enumerate(self.top_alleles.keys()):
            allele, freq = self.top_alleles[pos][0]

            if allele[1] == SNP and freq <= 2:
                continue

            start_index = self.positional_info_position_to_index[pos] - \
                          self.positional_info_position_to_index[ref_start]

            left_window_size = int(WINDOW_SIZE / 2)
            right_window_size = int(WINDOW_SIZE / 2) if WINDOW_SIZE % 2 == 0 else int(WINDOW_SIZE / 2) + 1
            left_window_index = start_index - left_window_size - CONTEXT_SIZE
            right_window_index = start_index + right_window_size + CONTEXT_SIZE

            if pos < interval_start - POS_BUFFER or pos > interval_end + POS_BUFFER:
                continue

            start_pos_is_insert = self.positional_info_index_to_position[start_index - left_window_size][1]
            start_pos = self.positional_info_index_to_position[start_index - left_window_size][0]
            if start_pos_is_insert:
                start_pos += 1

            end_pos = self.positional_info_index_to_position[start_index + right_window_size - 1][0]

            if end_pos < interval_start - POS_BUFFER or end_pos > interval_end + POS_BUFFER:
                continue

            if left_window_index < img_started_in_indx:
                continue
            if right_window_index > img_ended_in_indx:
                break

            img_left_index = left_window_index - img_started_in_indx
            img_right_index = right_window_index - img_started_in_indx
            label_left_index = start_index - left_window_size
            label_right_index = start_index + right_window_size

            sub_label_seq = label_seq[label_left_index:label_right_index]
            sub_ref_seq = ref_seq[img_left_index:img_right_index]

            hom_bases_count = collections.Counter(sub_label_seq)
            other_bases = sum(hom_bases_count.values()) - hom_bases_count['0']

            # if other_bases <= 0:
            #     continue
            #     include_this = True if random.random() < ALL_HOM_BASE_RATIO else False
            #     if not include_this:
            #         continue

            sliced_image = image[:, img_left_index:img_right_index, :]
            img_h, img_w, img_c = sliced_image.shape

            sliced_images.append(np.array(sliced_image, dtype=np.int8))
            index_info = str(image_index)
            sequence_info = str(self.chromosome_name) + " " + str(start_pos) + "," + str(sub_label_seq)
            sequence_info = sequence_info + "," + str(sub_ref_seq)
            summary_string = file_info + "," + index_info + "," + sequence_info + "\n"
            summary_strings = summary_strings + summary_string

            # print(pos, start_pos, end_pos)
            # from analysis.analyze_png_img import analyze_array
            # print(' ' * CONTEXT_SIZE + str(sub_label_seq))
            # analyze_array(sliced_image)
            # exit()
            image_index += 1

        return sliced_images, summary_strings, img_h, img_w, img_c

