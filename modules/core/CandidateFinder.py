from collections import defaultdict
from modules.handlers.BamHandler import BamHandler
from modules.handlers.FastaHandler import FastaHandler
import sys
from modules.handlers.TextColor import TextColor

"""
CandidateFinder module grabs all reads of a genomic region and processes each read individually.

This module populates dictionaries based on genomic positions that is later used to generate training tensors for
that specific region.
"""
# Reads below this mapping quality will not be processed
DEFAULT_MIN_MAP_QUALITY = 1
DEBUG_MESSAGE = False

# Indexing of each allele types [DO NOT CHANGE IF YOU DON'T KNOW WHAT YOU ARE DOING]
MATCH_ALLELE = 0
MISMATCH_ALLELE = 1
INSERT_ALLELE = 2
DELETE_ALLELE = 3

MIN_DELETE_QUALITY =20
BOUNDARY_COLUMNS = 20


class CandidateFinder:
    """
    Process a genomic region of a BAM file and populate dictionaries that can be used for tensor/image generation
    """
    def __init__(self, bam_file, ref_file, chr_name):
        """
        Initialize handlers and dictionaries
        :param bam_file: Path to BAM file
        :param ref_file: Path to reference file
        :param chr_name: Name of chromosome that needs to be processed
        """
        self.bam_handler = BamHandler(bam_file)
        self.fasta_handler = FastaHandler(ref_file)

        self.chromosome_name = chr_name

        # the base and the insert dictionary
        '''
        base_dictionary and insert_dictionary are used to store information about which base or insert allele is 
        present given a read's id. So, base_dictionary[read_id][genomic_position] should return (base, quality)
        which is the base and the base quality of that base in that genomic position for that read. Same works
        for insert dictionary, but it's a list of bases and qualities instead of a single base.
        
        insert_length_info saves the maximum length insert given a genomic position.
        '''
        self.base_dictionary = defaultdict(lambda: defaultdict(int))
        self.insert_dictionary = defaultdict(lambda: defaultdict(int))
        self.insert_length_info = defaultdict(int)

        # positional information
        '''
        positional_allele_frequency: Stores the frequency of (allele, type) in any position. Type in IN/DEL/SNP.
        reference_dictionary: Stores the base of reference for a given position
        coverage: keeps track of coverage for any given position
        rms_mq: mean root squared value of a mapping qualities of a position
        mismatch_count: count the number of allelic mismatches in a position
        read_info: stores read info (start, stop, mq, is_reverse) given read_id.
        allele_dictionary: list of all alleles in a position given read_id (allele, type)
        read_allele_dictionary: alleles found in a read while processing that read
        '''
        self.positional_allele_frequency = {}
        self.reference_dictionary = {}
        self.coverage = defaultdict(int)
        self.rms_mq = defaultdict(int)
        self.mismatch_count = defaultdict(int)
        self.read_info = defaultdict(list)
        self.allele_dictionary = defaultdict(lambda: defaultdict(list))
        self.read_allele_dictionary = {}

    @staticmethod
    def get_read_stop_position(read):
        """
        Returns the stop position of the reference to where the read stops aligning
        :param read: The read
        :return: stop position of the reference where the read last aligned
        """
        ref_alignment_stop = read.reference_end

        # only find the position if the reference end is fetched as none from pysam API
        if ref_alignment_stop is None:
            positions = read.get_reference_positions()

            # find last entry that isn't None
            i = len(positions) - 1
            ref_alignment_stop = positions[-1]
            while i > 0 and ref_alignment_stop is None:
                i -= 1
                ref_alignment_stop = positions[i]

        return ref_alignment_stop

    def _update_base_dictionary(self, read_id, pos, base, quality):
        """
        Update the base dictionary
        :param read_id: Unique read id
        :param pos: Genomic position
        :param base: Base in that genomic position for the read
        :param quality: Quality of the base
        :return:
        """
        self.base_dictionary[read_id][pos] = (base, quality)

    def _update_insert_dictionary(self, read_id, pos, bases, qualities):
        """
        Update the insert dictionary
        :param read_id: Unique read id
        :param pos: Genomic position
        :param bases: List of bases as insert
        :param qualities: List of qualities of those bases
        :return:
        """
        self.insert_dictionary[read_id][pos] = (bases, qualities)
        self.insert_length_info[pos] = max(self.insert_length_info[pos], len(bases))

    def _update_reference_dictionary(self, position, ref_base):
        """
        Update the reference dictionary
        :param position: Genomic position
        :param ref_base: Reference base at that position
        :return:
        """
        self.reference_dictionary[position] = ref_base

    def _update_read_allele_dictionary(self, read_id, pos, allele, allele_type):
        """
        Update the read dictionary with an allele
        :param pos: Genomic position
        :param allele: Allele found in that position
        :param allele_type: IN, DEL or SUB
        :return:
        """
        if pos not in self.read_allele_dictionary:
            self.read_allele_dictionary[pos] = {}
        if (allele, allele_type) not in self.read_allele_dictionary[pos]:
            self.read_allele_dictionary[pos][(allele, allele_type)] = 0

        self.read_allele_dictionary[pos][(allele, allele_type)] += 1

    def _update_positional_allele_frequency(self, read_id, pos, allele, allele_type):
        """
        Update the positional allele dictionary that contains whole genome allele information
        :param pos: Genomic position
        :param allele: Allele found
        :param allele_type: IN, DEL or SUB
        :return:
        """
        if pos not in self.positional_allele_frequency:
            self.positional_allele_frequency[pos] = {}
        if (allele, allele_type) not in self.positional_allele_frequency[pos]:
            self.positional_allele_frequency[pos][(allele, allele_type)] = 0

        # increase the allele frequency of the allele at that position
        self.positional_allele_frequency[pos][(allele, allele_type)] += 1
        self.allele_dictionary[read_id][pos].append((allele, allele_type))

    def parse_match(self, read_id, alignment_position, length, read_sequence, ref_sequence, qualities):
        """
        Process a cigar operation that is a match
        :param read_id: Unique read id
        :param alignment_position: Position where this match happened
        :param read_sequence: Read sequence
        :param ref_sequence: Reference sequence
        :param length: Length of the operation
        :param qualities: List of qualities for bases
        :return:

        This method updates the candidates dictionary.
        """
        start = alignment_position
        stop = start + length
        for i in range(start, stop):
            allele = read_sequence[i-alignment_position]
            ref = ref_sequence[i-alignment_position]

            self.coverage[i] += 1
            self._update_base_dictionary(read_id, i, allele, qualities[i-alignment_position])

            if allele != ref:
                self.mismatch_count[i] += 1
                self._update_read_allele_dictionary(read_id, i, allele, MISMATCH_ALLELE)

    def parse_delete(self, read_id, alignment_position, length, ref_sequence):
        """
        Process a cigar operation that is a delete
        :param alignment_position: Alignment position
        :param length: Length of the delete
        :param ref_sequence: Reference sequence of delete
        :return:

        This method updates the candidates dictionary.
        """
        # actual delete position starts one after the anchor
        start = alignment_position + 1
        stop = start + length
        self.mismatch_count[alignment_position] += 1

        for i in range(start, stop):
            self._update_base_dictionary(read_id, i, '*', MIN_DELETE_QUALITY)
            # increase the coverage
            self.mismatch_count[i] += 1
            self.coverage[i] += 1

        # the allele is the anchor + what's being deleted
        allele = self.reference_dictionary[alignment_position] + ref_sequence

        # record the delete where it first starts
        self._update_read_allele_dictionary(read_id, alignment_position + 1, allele, DELETE_ALLELE)

    def parse_insert(self, read_id, alignment_position, read_sequence, qualities):
        """
        Process a cigar operation where there is an insert
        :param alignment_position: Position where the insert happened
        :param read_sequence: The insert read sequence
        :return:

        This method updates the candidates dictionary. Mostly by adding read IDs to the specific positions.
        """
        # the allele is the anchor + what's being deleted
        allele = self.reference_dictionary[alignment_position] + read_sequence

        # record the insert where it first starts
        self.mismatch_count[alignment_position] += 1
        self._update_read_allele_dictionary(read_id, alignment_position + 1, allele, INSERT_ALLELE)
        self._update_insert_dictionary(read_id, alignment_position, read_sequence, qualities)

    def parse_cigar_tuple(self, cigar_code, length, alignment_position, ref_sequence, read_sequence, read_id, quality):
        """
        Parse through a cigar operation to find possible candidate variant positions in the read
        :param cigar_code: Cigar operation code
        :param length: Length of the operation
        :param alignment_position: Alignment position corresponding to the reference
        :param ref_sequence: Reference sequence
        :param read_sequence: Read sequence
        :param read_id: Unique read id
        :param quality: List of base qualities
        :return: read and reference index progress

        cigar key map based on operation.
        details: http://pysam.readthedocs.io/en/latest/api.html#pysam.AlignedSegment.cigartuples
        0: "MATCH",
        1: "INSERT",
        2: "DELETE",
        3: "REFSKIP",
        4: "SOFTCLIP",
        5: "HARDCLIP",
        6: "PAD"
        """
        # get what kind of code happened
        ref_index_increment = length
        read_index_increment = length

        # deal different kinds of operations
        if cigar_code == 0:
            # match
            self.parse_match(read_id=read_id,
                             alignment_position=alignment_position,
                             length=length,
                             read_sequence=read_sequence,
                             ref_sequence=ref_sequence,
                             qualities=quality)
        elif cigar_code == 1:
            # insert
            # alignment position is where the next alignment starts, for insert and delete this
            # position should be the anchor point hence we use a -1 to refer to the anchor point
            self.parse_insert(read_id=read_id,
                              alignment_position=alignment_position-1,
                              read_sequence=read_sequence,
                              qualities=quality)
            ref_index_increment = 0
        elif cigar_code == 2 or cigar_code == 3:
            # delete or ref_skip
            # alignment position is where the next alignment starts, for insert and delete this
            # position should be the anchor point hence we use a -1 to refer to the anchor point
            self.parse_delete(read_id=read_id,
                              alignment_position=alignment_position-1,
                              ref_sequence=ref_sequence,
                              length=length)
            read_index_increment = 0
        elif cigar_code == 4:
            # soft clip
            ref_index_increment = 0
            # print("CIGAR CODE ERROR SC")
        elif cigar_code == 5:
            # hard clip
            ref_index_increment = 0
            read_index_increment = 0
            # print("CIGAR CODE ERROR HC")
        elif cigar_code == 6:
            # pad
            ref_index_increment = 0
            read_index_increment = 0
            # print("CIGAR CODE ERROR PAD")
        else:
            raise("INVALID CIGAR CODE: %s" % cigar_code)

        return ref_index_increment, read_index_increment

    def process_read(self, read, interval_start, interval_end):
        """
        Find candidate in reads an populate the dictionaries
        :param read: Read from which we need to find the variant candidate positions.
        :param interval_start: start position of the genomic interval
        :param interval_end: end position of the the genomic interval
        :return:
        """
        # clear the read allele dictionary
        # we populate this dictionary for each read
        self.read_allele_dictionary = {}
        # positions where the read start and stop aligning to the reference
        ref_alignment_start = read.reference_start
        ref_alignment_stop = self.get_read_stop_position(read)
        # cigar tuples
        cigar_tuples = read.cigartuples
        # the read sequence
        read_sequence = read.query_sequence
        # unique read id, has to be unique
        read_id = read.query_name
        # base qualities of the read
        read_quality = read.query_qualities

        # reference sequence of the read
        ref_sequence = self.fasta_handler.get_sequence(chromosome_name=self.chromosome_name,
                                                       start=ref_alignment_start,
                                                       stop=ref_alignment_stop+10)

        # save the read information
        self.read_info[read_id] = (ref_alignment_start, ref_alignment_stop, read.mapping_quality, read.is_reverse)

        # update the reference dictionary
        for i, ref_base in enumerate(ref_sequence):
            self._update_reference_dictionary(ref_alignment_start + i, ref_base)

        # read_index: index of read sequence
        # ref_index: index of reference sequence
        read_index = 0
        ref_index = 0

        # until we first find a match we won't process that read. All preceding inserts and deletes are discarded.
        found_valid_cigar = False

        # per cigar operation
        for cigar in cigar_tuples:
            # cigar code
            cigar_code = cigar[0]
            # length of the cigar operation
            length = cigar[1]

            # get the sequence segments that are effected by this operation
            ref_sequence_segment = ref_sequence[ref_index:ref_index+length]
            read_quality_segment = read_quality[read_index:read_index+length]
            read_sequence_segment = read_sequence[read_index:read_index+length]

            if cigar_code != 0 and found_valid_cigar is False:
                read_index += length
                continue
            found_valid_cigar = True

            # send the cigar tuple to get attributes we got by this operation
            ref_index_increment, read_index_increment = \
                self.parse_cigar_tuple(cigar_code=cigar_code,
                                       length=length,
                                       alignment_position=ref_alignment_start+ref_index,
                                       ref_sequence=ref_sequence_segment,
                                       read_sequence=read_sequence_segment,
                                       read_id=read_id,
                                       quality=read_quality_segment)

            # increase the read index iterator
            read_index += read_index_increment
            ref_index += ref_index_increment

        # after collecting all alleles from reads, update the global dictionary
        for position in self.read_allele_dictionary.keys():
            if position < interval_start or position > interval_end:
                continue
            self.rms_mq[position] += read.mapping_quality * read.mapping_quality
            for record in self.read_allele_dictionary[position]:
                # there can be only one record per position in a read
                allele, allele_type = record

                if allele_type == MISMATCH_ALLELE:
                    # If next allele is indel then group it with the current one, don't make a separate one
                    if position + 1 <= ref_alignment_stop and position + 1 in self.read_allele_dictionary.keys():
                        next_allele, next_allele_type = list(self.read_allele_dictionary[position + 1].keys())[0]
                        if next_allele_type == INSERT_ALLELE or next_allele_type == DELETE_ALLELE:
                            continue
                    self._update_positional_allele_frequency(read_id, position, allele, allele_type)
                else:
                    # it's an insert or delete, so, add to the previous position
                    self._update_positional_allele_frequency(read_id, position - 1, allele, allele_type)

    def process_interval(self, interval_start, interval_end):
        """
        Processes all reads and reference for this interval
        :param interval_start: start position of a genomic interval
        :param interval_end: end position of a genomic interval
        :return:
        """
        # get all the reads of that region
        reads = self.bam_handler.get_reads(self.chromosome_name, interval_start - BOUNDARY_COLUMNS,
                                           interval_end + BOUNDARY_COLUMNS)

        total_reads = 0
        read_id_list = []
        for read in reads:
            # check if the read is usable
            if read.mapping_quality >= DEFAULT_MIN_MAP_QUALITY and read.is_secondary is False \
                    and read.is_supplementary is False and read.is_unmapped is False and read.is_qcfail is False:
                # for paired end make sure read name is unique
                read.query_name = read.query_name + '_1' if read.is_read1 else read.query_name + '_2'
                self.process_read(read, interval_start - BOUNDARY_COLUMNS, interval_end + BOUNDARY_COLUMNS)
                read_id_list.append(read.query_name)
                total_reads += 1

        if DEBUG_MESSAGE:
            sys.stderr.write(TextColor.BLUE)
            sys.stderr.write("INFO: TOTAL READ IN REGION: " + self.chromosome_name + " " + str(interval_start) + " " +
                             str(interval_end) + " " + str(total_reads) + "\n" + TextColor.END)

        return read_id_list
