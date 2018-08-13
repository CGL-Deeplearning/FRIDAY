MAX_COLOR_VALUE = 254.0
BASE_QUALITY_CAP = 40.0
MAP_QUALITY_CAP = 60.0
MAP_QUALITY_FILTER = 5.0
MIN_DELETE_QUALITY = 20.0
MATCH_CIGAR_CODE = 0
INSERT_CIGAR_CODE = 1
DELETE_CIGAR_CODE = 2
IMAGE_DEPTH_THRESHOLD = 120
HIGHEST_ALLELE_LENGTH = 10

global_ref_base_values = {'A': 25.0, 'C': 75.0, 'G': 125.0, 'T': 175.0, '*': 225.0, 'N': 10.0}
global_read_base_values = {'A': 0.0, 'C': 5.0, 'G': 10.0, 'T': 15.0, '*': 20.0, '-': 25.0}


class ImageChannels:
    """
    Handles how many channels to create for each base and their way of construction.
    """

    def __init__(self, pileup_attributes, ref_base):
        """
        Initialize a base with it's attributes
        :param pileup_attributes: Attributes of a pileup base
        :param ref_base: Reference base corresponding to that pileup base
        """
        self.base_quality = pileup_attributes[1]
        self.map_quality = pileup_attributes[2]
        self.cigar_code = pileup_attributes[3]
        self.pileup_base = pileup_attributes[0] if self.cigar_code != DELETE_CIGAR_CODE else '-'
        self.is_rev = pileup_attributes[4]
        self.is_duplicate = pileup_attributes[5]
        self.is_qc_fail = pileup_attributes[6]
        self.is_read1 = pileup_attributes[7]
        self.is_read2 = pileup_attributes[8]
        self.is_mate_reverse = pileup_attributes[9]
        self.allele_length = pileup_attributes[10]
        self.support_allele = pileup_attributes[11]
        self.ref_base = ref_base

    @staticmethod
    def get_total_number_of_channels():
        return len(ImageChannels.get_empty_channels())

    @staticmethod
    def get_empty_channels():
        """
        Get empty channel values
        :return:
        """
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def get_channels(self):
        """
        Get a bases's channel construction
        :return: [color spectrum of channels based on base attributes]
        """
        if self.pileup_base == self.ref_base or self.pileup_base == 'N' or self.ref_base == 'N'\
                or self.ref_base not in global_ref_base_values or self.pileup_base not in global_read_base_values:
            base_difference = 5.0
        else:
            base_difference = global_ref_base_values[self.ref_base] + global_read_base_values[self.pileup_base]

        base_quality = (MAX_COLOR_VALUE * min(self.base_quality, BASE_QUALITY_CAP)) / BASE_QUALITY_CAP

        map_quality = (MAX_COLOR_VALUE * min(self.map_quality, MAP_QUALITY_CAP)) / MAP_QUALITY_CAP

        is_reverse = 254.0 if self.is_rev else 70.0

        is_duplicate = 5.0 if self.is_duplicate else 254.0

        is_qc_fail = 5.0 if self.is_qc_fail else 254.0

        is_read1_or_2 = 254.0 if self.is_read1 else 150.0

        is_mate_reverse = 125.0 if self.is_mate_reverse else 254.0

        allele_length = 0
        if self.allele_length == 1:
            allele_length = 150.0
        elif self.allele_length > 1:
            allele_length = 254.0

        support_allele = 254.0
        if self.support_allele == 2:
            support_allele = 125.0
        elif self.support_allele == 0:
            support_allele = 0.0

        return [base_difference, base_quality, map_quality, is_reverse,
                is_duplicate, is_qc_fail, is_read1_or_2, is_mate_reverse, allele_length, support_allele]

    @staticmethod
    def get_channels_for_ref(ref_base):
        """
        Get a reference bases's channel construction
        :param ref_base: Reference base
        :return: [color spectrum of channels based on some default values]
        """
        base_difference = global_ref_base_values[ref_base] if ref_base in global_read_base_values else 5.0
        # maximum base_quality
        base_quality = MAX_COLOR_VALUE
        # maximum mapping quality
        map_quality = MAX_COLOR_VALUE
        # not reverse
        is_reverse = 70.0
        # not duplicate
        is_duplicate = 254.0
        # not qc failed
        is_qc_fail = 254.0
        # always read1
        is_read1_or_2 = 250.0
        # mate is not reversed
        is_mate_reverse = 254.0
        support_allele = 254.0

        # consider allele length to be 1
        allele_length = HIGHEST_ALLELE_LENGTH
        allele_length = (MAX_COLOR_VALUE * min(allele_length, HIGHEST_ALLELE_LENGTH)) / HIGHEST_ALLELE_LENGTH

        return [base_difference, base_quality, map_quality, is_reverse,
                is_duplicate, is_qc_fail, is_read1_or_2, is_mate_reverse, allele_length, support_allele]
