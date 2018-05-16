from collections import defaultdict
from modules.handlers.ImageChannels import ImageChannels
from scipy import misc
import sys
import numpy as np
from modules.handlers.TextColor import TextColor

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
WINDOW_OVERLAP_JUMP = 249
# image size
WINDOW_SIZE = 300
# flanking size is the amount add on each size
WINDOW_FLANKING_SIZE = 5
# boundary columns is the number of bases we process for safety
BOUNDARY_COLUMNS = 5

# Logging configuration
LOG_LEVEL_HIGH = 1
LOG_LEVEL_LOW = 0
LOG_LEVEL = LOG_LEVEL_LOW
WARN_COLOR = TextColor.RED


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
        self.image_row_for_reads = defaultdict(list)
        self.image_row_for_ref = defaultdict(list)
        self.positional_info_index_to_position = defaultdict(int)
        self.positional_info_position_to_index = defaultdict(int)
        self.base_frequency = defaultdict(lambda: defaultdict(int))
        self.index_based_coverage = defaultdict(int)
        self.reference_base_by_index = defaultdict(int)
        self.vcf_positional_dict = defaultdict(list)
        self.reference_string = ''

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
            for pos in range(start_pos, end_pos):

                if pos < interval_start:
                    continue

                if pos > interval_end:
                    break

                if pos not in self.pos_dicts.base_dictionary[read_id] and \
                        pos not in self.pos_dicts.insert_dictionary[read_id]:
                    print(pos, read_id)
                    continue

                # if there is a base in that position for that read
                if pos in self.pos_dicts.base_dictionary[read_id]:
                    # get the base and the base quality
                    base, base_q = self.pos_dicts.base_dictionary[read_id][pos]
                    # see if the base is a delete
                    cigar_code = 0 if base != '*' else 1
                    # get the reference base of that position
                    ref_base = self.pos_dicts.reference_dictionary[pos]
                    # combine all the pileup attributes we want to encode in the image
                    pileup_attributes = (base, base_q, mapping_quality, cigar_code, strand_direction)
                    # create a channel object to covert these features to a pixel
                    channel_object = ImageChannels(pileup_attributes, ref_base, 0)
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
                            pileup_attributes = (base, base_q, mapping_quality, cigar_code, strand_direction)
                            channel_object = ImageChannels(pileup_attributes, ref_base, 0)
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
                            pileup_attributes = (base, base_q, mapping_quality, cigar_code, strand_direction)
                            channel_object = ImageChannels(pileup_attributes, ref_base, 0)
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

    def get_allele_bases_from_vcf_genotype(self, indx, vcf_records, base_frequencies, ref_base):
        """
        Get allelic bases using vcf
        :param indx: Index of corresponding position
        :param vcf_records: VCF records of that position
        :param base_frequencies: Base frequency dictionary of the position
        :param ref_base: Reference base
        :return:
        """
        bases = []
        for vcf_record in vcf_records:
            allele, genotype = vcf_record
            if not base_frequencies[allele] and LOG_LEVEL == LOG_LEVEL_HIGH:
                vcf_record_msg = str(self.pos_dicts.chromosome_name) + "\t" + str(vcf_record)
                warn_msg = "WARN:\tPOSITIONAL MISMATCH\t" + str(indx) + "\t" + vcf_record_msg + "\n"
                sys.stderr.write(WARN_COLOR + warn_msg + TextColor.END)
            # hom_alt
            if genotype[0] == genotype[1]:
                bases.append(allele)
                bases.append(allele)
            else:
                bases.append(allele)
        if len(bases) == 0:
            return ref_base, ref_base
        elif len(bases) == 1:
            return bases[0], ref_base
        else:
            return bases[0], bases[1]

    @staticmethod
    def get_translated_letter(base_a, base_b):
        """
        Given two bases, translate them to a value for labeling
        :param base_a: Base 1
        :param base_b: Base 2
        :return:
        """
        if base_b < base_a:
            base_a, base_b = base_b, base_a
        encoded_base = base_a + base_b
        encoded_base = encoded_base.replace('N', '.')
        base_to_letter_map = {'**': 0, '*.': 1, '*A': 2, '*C': 3, '*G': 4, '*T': 5,
                                       '..': 6, '.A': 7, '.C': 8, '.G': 9, '.T': 10,
                                                'AA': 11, 'AC': 12, 'AG': 13, 'AT': 14,
                                                          'CC': 15, 'CG': 16, 'CT': 17,
                                                                    'GG': 18, 'GT': 19,
                                                                              'TT': 20}
        return base_to_letter_map[encoded_base]

    def get_label_sequence(self, interval_start, interval_end):
        """
        Get the label of a genomic interval
        :param interval_start: Interval start
        :param interval_end: Interval end
        :return:
        """
        start_index = self.positional_info_position_to_index[interval_start]
        end_index = self.positional_info_position_to_index[interval_end]
        # we try to pick bases naively based on frequencies for debugging purpose
        string_a = ''
        string_b = ''
        # the bases we see in VCF
        vcf_string_a = ''
        vcf_string_b = ''
        # helps keep track of positions
        positional_values = ''
        # label sequence
        translated_sequence = ''
        reference_string = ''

        for i in range(start_index, end_index):
            # build the reference string
            reference_string += self.reference_string[i]
            # see if we are in insert or in a true genomic position
            pos_increase = 0 if self.positional_info_index_to_position[i][1] is True else 1
            # reference base
            ref_base = self.reference_base_by_index[i]
            positional_values = positional_values + str(pos_increase)
            vcf_alts = []
            # from the positional vcf upte the label
            if i in self.vcf_positional_dict:
                alt_a, alt_b = self.get_allele_bases_from_vcf_genotype(i, self.vcf_positional_dict[i],
                                                                       self.base_frequency[i], ref_base)
                vcf_string_a += alt_a
                vcf_string_b += alt_b
                vcf_alts.append(alt_a)
                vcf_alts.append(alt_b)
                translated_sequence = translated_sequence + chr(ord('A') + self.get_translated_letter(alt_a, alt_b))
            else:
                vcf_string_a += ref_base
                vcf_string_b += ref_base
                translated_sequence = translated_sequence + chr(ord('A') + self.get_translated_letter(ref_base, ref_base))

            # naively pick most frequent bases and see if you are way off
            # helpful for debugging
            bases = []
            total_bases = self.index_based_coverage[i]
            for base in self.base_frequency[i]:
                base_frequency = self.base_frequency[i][base] / total_bases
                if base_frequency >= ALLELE_FREQUENCY_THRESHOLD_FOR_REPORTING:
                    bases.append(base)
                    # warn if a really abundant allele is not in VCF
                    if base not in vcf_alts and base != ref_base and LOG_LEVEL == LOG_LEVEL_HIGH:
                        genome_position = self.positional_info_index_to_position[i][0]
                        msg = str(self.chromosome_name) + "\t" + str(genome_position) + "\t" + str(ref_base) + "\t" \
                              + str(base) + "\t" + str(self.base_frequency[i][base]) + "\t" + str(total_bases) + "\t" \
                              + str(int(base_frequency*100)) + "\n"
                        warn_msg = "WARN:\tBAM MISMATCH\t" + msg
                        sys.stderr.write(WARN_COLOR + warn_msg + TextColor.END)
            if len(bases) == 0:
                string_a += '-'
                string_b += '-'
            elif len(bases) == 1:
                string_a += bases[0] if bases[0] != ref_base else '-'
                string_b += bases[0] if bases[0] != ref_base else '-'
            else:
                string_a += bases[0] if bases[0] != ref_base else '-'
                string_b += bases[1] if bases[1] != ref_base else '-'

        if ALLELE_DEBUG:
            print(string_a)
            print(string_b)

        return vcf_string_a, vcf_string_b, translated_sequence, positional_values, reference_string

    def populate_vcf_alleles(self, positional_vcf, interval_start, interval_end):
        """
        From positional VCF alleles, populate the positional dictionary.
        :param positional_vcf: Positional VCF dictionar
        :param interval_start: Interval start
        :param interval_end: Interval end
        :return:
        """
        for pos in positional_vcf.keys():
            if pos < interval_start or pos > interval_end:
                continue
            # get bam position
            bam_pos = pos + VCF_INDEX_BUFFER
            indx = self.positional_info_position_to_index[bam_pos]

            snp_recs, in_recs, del_recs = positional_vcf[pos]
            alt_alleles_found = self.pos_dicts.positional_allele_frequency[bam_pos] \
                if bam_pos in self.pos_dicts.positional_allele_frequency else []

            # SNP records
            for snp_rec in snp_recs:
                alt_allele = snp_rec.alt[0]
                alt_ = (alt_allele, 1)
                # check if the allele is actually present in the BAM
                if alt_ in alt_alleles_found:
                    self.vcf_positional_dict[indx].append((alt_allele, snp_rec.genotype))
                elif LOG_LEVEL == LOG_LEVEL_HIGH:
                    vcf_record_msg = str(self.chromosome_name) + "\t" + str(snp_rec)
                    warn_msg = "WARN:\tVCF MISMATCH\t" + str(indx) + "\t" + vcf_record_msg + "\n"
                    sys.stderr.write(WARN_COLOR + warn_msg + TextColor.END)

            # insert record
            for in_rec in in_recs:
                alt_ = (in_rec.alt, 2)
                # check if the allele is actually present in the BAM
                if alt_ in alt_alleles_found:
                    for i in range(1, len(in_rec.alt)):
                        self.vcf_positional_dict[indx+i].append((in_rec.alt[i], in_rec.genotype))
                elif LOG_LEVEL == LOG_LEVEL_HIGH:
                    vcf_record_msg = str(self.chromosome_name) + "\t" + str(in_rec)
                    warn_msg = "WARN:\tVCF MISMATCH\t" + str(indx) + "\t" + vcf_record_msg + "\n"
                    sys.stderr.write(WARN_COLOR + warn_msg + TextColor.END)

            # delete record
            for del_rec in del_recs:
                alt_ = (del_rec.ref, 3)
                # check if the allele is actually present in the BAM
                if alt_ in alt_alleles_found:
                    for i in range(1, len(del_rec.ref)):
                        del_indx = self.positional_info_position_to_index[bam_pos+i]
                        self.vcf_positional_dict[del_indx].append(('.', del_rec.genotype))
                elif LOG_LEVEL == LOG_LEVEL_HIGH:
                    vcf_record_msg = str(self.chromosome_name) + "\t" + str(del_rec)
                    warn_msg = "WARN:\tVCF MISMATCH\t" + str(indx) + "\t" + vcf_record_msg + "\n"
                    sys.stderr.write(WARN_COLOR + warn_msg + TextColor.END)

    def create_region_alignment_image(self, interval_start, interval_end, positional_variants, read_id_list):
        """
        Generate labeled images of a given region of the genome
        :param interval_start: Starting genomic position of the interval
        :param interval_end: End genomic position of the interval
        :param positional_variants: List of positional variants in that region
        :param read_id_list: List of reads ids that fall in this region
        :return:
        """
        image = self.create_image(interval_start - BOUNDARY_COLUMNS, interval_end + BOUNDARY_COLUMNS, read_id_list)
        self.populate_vcf_alleles(positional_variants, interval_start, interval_end)
        vcf_a, vcf_b, translated_seq, pos_vals, ref_seq = self.get_label_sequence(interval_start - BOUNDARY_COLUMNS,
                                                                                  interval_end + BOUNDARY_COLUMNS)

        return image, vcf_a, vcf_b, translated_seq, pos_vals, ref_seq

    def get_segmented_image_sequences(self, interval_start, interval_end, positional_variants, read_id_list):
        """
        Generates segmented image sequences for training
        :param interval_start: Genomic interval start
        :param interval_end: Genomic interval stop
        :param positional_variants: VCF positional variants
        :param read_id_list: List of reads ids that fall in this region
        :return:
        """
        # post process reference and read
        self.post_process_reference(interval_start - BOUNDARY_COLUMNS, interval_end + BOUNDARY_COLUMNS)
        self.post_process_reads(read_id_list, interval_start - BOUNDARY_COLUMNS, interval_end + BOUNDARY_COLUMNS)

        image, vcf_a, vcf_b, translated_seq, pos_vals, ref_seq = \
            self.create_region_alignment_image(interval_start, interval_end, positional_variants, read_id_list)
        sliced_windows = []
        ref_row, ref_start, ref_end = self.image_row_for_ref
        img_started_in_indx = self.positional_info_position_to_index[interval_start - BOUNDARY_COLUMNS] - \
                              self.positional_info_position_to_index[ref_start]

        img_ended_in_indx = self.positional_info_position_to_index[interval_end + BOUNDARY_COLUMNS] - \
                            self.positional_info_position_to_index[ref_start]

        for pos in range(interval_start, interval_end, WINDOW_OVERLAP_JUMP):
            point_indx = self.positional_info_position_to_index[pos] - \
                         self.positional_info_position_to_index[ref_start]
            left_window_index = point_indx - WINDOW_FLANKING_SIZE
            right_window_index = point_indx + WINDOW_SIZE + WINDOW_FLANKING_SIZE

            if left_window_index < img_started_in_indx:
                continue
            if right_window_index > img_ended_in_indx:
                break
            img_left_indx = left_window_index - img_started_in_indx
            img_right_indx = right_window_index - img_started_in_indx
            sub_translated_seq = translated_seq[img_left_indx:img_right_indx]
            sub_pos_vals = pos_vals[img_left_indx:img_right_indx]
            sub_ref_seq = ref_seq[img_left_indx:img_right_indx]
            sliced_windows.append((pos, img_left_indx, img_right_indx, sub_translated_seq, sub_pos_vals,
                                   sub_ref_seq))

        return image, sliced_windows
