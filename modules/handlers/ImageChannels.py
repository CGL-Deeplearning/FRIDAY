MAX_COLOR_VALUE = 254.0
BASE_QUALITY_CAP = 40.0
MAP_QUALITY_CAP = 60.0
MAP_QUALITY_FILTER = 5.0
MIN_DELETE_QUALITY = 20.0
MATCH_CIGAR_CODE = 0
INSERT_CIGAR_CODE = 1
DELETE_CIGAR_CODE = 2
IMAGE_DEPTH_THRESHOLD = 300

global_base_color_dictionary = {'A': 254.0, 'C': 100.0, 'G': 180.0, 'T': 30.0, '*': 5.0, '.': 5.0, 'N': 5.0}
global_cigar_color_dictionary = {0: MAX_COLOR_VALUE, 1: MAX_COLOR_VALUE*0.6, 2: MAX_COLOR_VALUE*0.3}


class ImageChannels:
    """
    Handles how many channels to create for each base and their way of construction.
    """

    def __init__(self, pileup_attributes, ref_base, is_supporting):
        """
        Initialize a base with it's attributes
        :param pileup_attributes: Attributes of a pileup base
        :param ref_base: Reference base corresponding to that pileup base
        """
        self.pileup_base = pileup_attributes[0]
        self.base_qual = pileup_attributes[1]
        self.map_qual = pileup_attributes[2]
        self.cigar_code = pileup_attributes[3]
        self.is_rev = pileup_attributes[4]
        self.ref_base = ref_base
        self.is_match = True if self.ref_base == self.pileup_base else False

    @staticmethod
    def get_total_number_of_channels():
        return len(ImageChannels.get_empty_channels())

    @staticmethod
    def get_empty_channels():
        """
        Get empty channel values
        :return:
        """
        return [0, 0, 0, 0, 0, 0]

    def get_channels(self):
        """
        Get a bases's channel construction
        :return: [color spectrum of channels based on base attributes]
        """
        base_color = global_base_color_dictionary[self.pileup_base] \
            if self.pileup_base in global_base_color_dictionary else 0.0

        base_quality_color = (MAX_COLOR_VALUE * min(self.base_qual, BASE_QUALITY_CAP)) / BASE_QUALITY_CAP

        map_quality_color = (MAX_COLOR_VALUE * min(self.map_qual, MAP_QUALITY_CAP)) / MAP_QUALITY_CAP

        strand_color = 240.0 if self.is_rev else 70.0

        match_color = MAX_COLOR_VALUE * 0.2 if self.is_match is True else MAX_COLOR_VALUE * 1.0

        cigar_color = global_cigar_color_dictionary[self.cigar_code] \
            if self.cigar_code in global_cigar_color_dictionary else 0.0

        return [base_color, base_quality_color, map_quality_color, strand_color, match_color, cigar_color]

    @staticmethod
    def get_channels_for_ref(pileup_base):
        """
        Get a reference bases's channel construction
        :param pileup_base: Reference base
        :return: [color spectrum of channels based on some default values]
        """
        cigar_code = MATCH_CIGAR_CODE if pileup_base != '*' else INSERT_CIGAR_CODE
        base_qual = BASE_QUALITY_CAP
        map_qual = 60.0
        is_rev = False
        is_match = True

        base_color = global_base_color_dictionary[pileup_base] \
            if pileup_base in global_base_color_dictionary else 0.0

        base_quality_color = (MAX_COLOR_VALUE * min(base_qual, BASE_QUALITY_CAP)) / BASE_QUALITY_CAP

        map_quality_color = (MAX_COLOR_VALUE * min(map_qual, MAP_QUALITY_CAP)) / MAP_QUALITY_CAP

        strand_color = 240.0 if is_rev else 70.0

        match_color = MAX_COLOR_VALUE * 0.2 if is_match is True else MAX_COLOR_VALUE * 1.0

        cigar_color = global_cigar_color_dictionary[cigar_code] \
            if cigar_code in global_cigar_color_dictionary else 0.0

        return [base_color, base_quality_color, map_quality_color, strand_color, match_color, cigar_color]
