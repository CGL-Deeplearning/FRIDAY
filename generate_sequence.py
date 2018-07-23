import argparse
import time
import os
import sys
import multiprocessing
import pickle
import h5py
import numpy as np
from tqdm import tqdm

from modules.core.CandidateFinder import CandidateFinder
from modules.core.ImageGenerator import ImageGenerator
from modules.handlers.BamHandler import BamHandler
from modules.handlers.FastaHandler import FastaHandler
from modules.handlers.TextColor import TextColor
from modules.handlers.TsvHandler import TsvHandler
from modules.handlers.VcfHandler import VCFFileProcessor
from modules.handlers.FileManager import FileManager
"""
This script creates training sequences from BAM, Reference FASTA and truth VCF file. The process is:
- Find candidates that can be variants
- Label candidates using the VCF
- Create images for each candidate

Input:
- BAM file: Alignment of a genome
- REF file: The reference FASTA file used in the alignment
- VCF file: A truth VCF file
- BED file: A confident bed file. If confident_bed is passed it will only generate train set for those region.

Output:
- PNG files: Images that can be converted to tensors.
- CSV file: Containing records of images and their location in the H5PY file.
"""

# Global debug helpers
DEBUG_PRINT_CANDIDATES = False
DEBUG_TIME_PROFILE = False
DEBUG_TEST_PARALLEL = False
LOG_LEVEL_HIGH = 0
LOG_LEVEL_LOW = 1
LOG_LEVEL = LOG_LEVEL_LOW

TRAIN_MODE = 1
TEST_MODE = 2


MIN_SEQUENCE_BASE_LENGTH_THRESHOLD = 0
MIN_VARIANT_IN_WINDOW_THRESHOLD = 0
BED_INDEX_BUFFER = -1
SAFE_BOUNDARY_BASES = 50


def build_chromosomal_interval_trees(confident_bed_path):
    """
    Produce a dictionary of intervals trees, with one tree per chromosome
    :param confident_bed_path: Path to confident bed file
    :return: trees_chromosomal
    """
    # create an object for tsv file handling
    tsv_handler_reference = TsvHandler(tsv_file_path=confident_bed_path)
    # create intervals based on chromosome
    intervals_chromosomal_reference = tsv_handler_reference.get_bed_intervals_by_chromosome(start_offset=1,
                                                                                            universal_offset=-1)
    # create a dictionary to get all chromosomal trees
    intervals_chromosomal = dict()

    # for each chromosome extract the tree and add it to the dictionary
    for chromosome_name in intervals_chromosomal_reference:
        intervals = intervals_chromosomal_reference[chromosome_name]
        intervals_chromosomal[chromosome_name] = intervals

    # return the dictionary containing all the trees
    return intervals_chromosomal


class View:
    """
    Process manager that runs sequence of processes to generate images and their labebls.
    """
    def __init__(self, chromosome_name, bam_file_path, reference_file_path, vcf_path, output_file_path,
                 confidence_intervals, thread_no):
        """
        Initialize a manager object
        :param chromosome_name: Name of the chromosome
        :param bam_file_path: Path to the BAM file
        :param reference_file_path: Path to the reference FASTA file
        :param vcf_path: Path to the VCF file
        :param output_file_path: Path to the output directory where images are saved
        :param confidence_intervals: Confidence interval in the chromosome.
        """
        # --- initialize handlers ---
        # create objects to handle different files and query
        self.bam_handler = BamHandler(bam_file_path)
        self.fasta_handler = FastaHandler(reference_file_path)
        self.output_dir = output_file_path
        self.confidence_intervals = confidence_intervals
        self.vcf_path = vcf_path
        self.candidate_finder = CandidateFinder(bam_file_path, reference_file_path, chromosome_name)
        self.thread_id = thread_no

        # --- initialize names ---
        # name of the chromosome
        self.chromosome_name = chromosome_name
        self.summary_file = open(self.output_dir + "summary/" + "summary" + '_' + chromosome_name + "_" +
                                 str(thread_no) + ".csv", 'w')

    def get_vcf_record_of_region(self, start_pos, end_pos, filter_hom_ref=False):
        """
        Get VCF records of a given region
        :param start_pos: start position of the region
        :param end_pos: end position of the region
        :param filter_hom_ref: whether to ignore hom_ref VCF records during candidate validation
        :return: VCF_records: VCF records of the region
        """

        vcf_handler = VCFFileProcessor(file_path=self.vcf_path)
        # get dictionary of variant records for full region
        vcf_handler.populate_dictionary(contig=self.chromosome_name,
                                        start_pos=start_pos,
                                        end_pos=end_pos,
                                        hom_filter=filter_hom_ref)

        # get separate positional variant dictionaries for IN, DEL, and SNP
        positional_variants = vcf_handler.get_variant_dictionary()

        return positional_variants

    @staticmethod
    def save_dictionary(dictionary, file_path):
        """
        Save a dictionary to a file.
        :param dictionary: The dictionary to save
        :param file_path: Path to the output file
        :return:
        """
        with open(file_path, 'wb') as f:
            pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)

    def parse_region(self, start_index, end_index, test_mode=True):
        """
        Generate labeled images of a given region of the genome
        :param start_index: Start index of the confident interval
        :param end_index: End index of the confident interval
        :return:
        """
        for i in range(start_index, end_index):
            interval_start, interval_end = self.confidence_intervals[i][0] + BED_INDEX_BUFFER, \
                                           self.confidence_intervals[i][1] + BED_INDEX_BUFFER

            interval_length = interval_end - interval_start

            if interval_length < 300 and test_mode is True:
                diff = 300 - interval_length + 10
                interval_start = interval_start - (diff + 10)
                interval_end = interval_end + (diff + 10)

            if interval_length < MIN_SEQUENCE_BASE_LENGTH_THRESHOLD:
                warn_msg = "REGION SKIPPED, TOO SMALL OF A WINDOW " + self.chromosome_name + " "
                warn_msg = warn_msg + str(interval_start) + " " + str(interval_end) + "\n"
                if LOG_LEVEL == LOG_LEVEL_HIGH:
                    sys.stderr.write(TextColor.BLUE + "INFO: " + warn_msg + TextColor.END)
                continue

            # create a h5py file where the images are stored
            hdf5_filename = os.path.abspath(self.output_dir) + '/' + str(self.chromosome_name) + '_' \
                            + str(interval_start) + ".h5"

            allele_dict_filename = self.chromosome_name + '_' + str(interval_start) + '_' + str(interval_end)
            allele_dict_filename = os.path.abspath(self.output_dir) + "/candidate_dictionaries/" \
                                   + allele_dict_filename + '.pkl'

            file_info = hdf5_filename + " " + allele_dict_filename

            # get positional variants
            positional_variants = self.get_vcf_record_of_region(interval_start - SAFE_BOUNDARY_BASES,
                                                                interval_end + SAFE_BOUNDARY_BASES)

            if len(positional_variants) < MIN_VARIANT_IN_WINDOW_THRESHOLD:
                warn_msg = "REGION SKIPPED, INSUFFICIENT NUMBER OF VARIANTS " + self.chromosome_name + " "
                warn_msg = warn_msg + str(interval_start) + " " + str(interval_end) + " VARIANT COUNT: " + \
                           str(len(positional_variants)) + "\n"
                if LOG_LEVEL == LOG_LEVEL_HIGH:
                    sys.stderr.write(TextColor.BLUE + "INFO: " + warn_msg + TextColor.END)
                continue

            # process the interval and populate dictionaries
            read_id_list = self.candidate_finder.process_interval(interval_start - SAFE_BOUNDARY_BASES,
                                                                  interval_end + SAFE_BOUNDARY_BASES)

            image_generator = ImageGenerator(self.candidate_finder)
            # get trainable sequences
            sliced_images, summary_strings, img_h, img_w, img_c = \
                image_generator.get_segmented_image_sequences(interval_start, interval_end, positional_variants,
                                                              read_id_list, file_info)

            if len(summary_strings) > 0:
                # save allele dictionary
                allele_dictionary = image_generator.top_alleles
                self.save_dictionary(allele_dictionary, allele_dict_filename)

                self.summary_file.write(summary_strings)

                hdf5_file = h5py.File(hdf5_filename, mode='w')
                # the image dataset we save. The index name in h5py is "images".
                img_dset = hdf5_file.create_dataset("images", (len(sliced_images),) + (img_h, img_w, img_c), np.uint8,
                                                    compression='gzip')
                # save the images and labels to the h5py file
                img_dset[...] = sliced_images


def test(view_object):
    """
    Run a test
    :return:
    """
    start_time = time.time()
    view_object.parse_region(start_index=0, end_index=5)
    print("TOTAL TIME ELAPSED: ", time.time()-start_time)


def parallel_run(chr_name, bam_file, ref_file, vcf_file, output_dir, conf_intervals, thread_no):
    """
    Creates a view object for a region and generates images for that region.
    :param chr_name: Name of the chromosome
    :param bam_file: path to BAM file
    :param ref_file: path to reference FASTA file
    :param vcf_file: path to VCF file
    :param output_dir: path to output directory
    :param conf_intervals: list containing confident bed intervals
    :param thread_no: thread number
    :return:
    """

    # create a view object
    view_ob = View(chromosome_name=chr_name,
                   bam_file_path=bam_file,
                   reference_file_path=ref_file,
                   output_file_path=output_dir,
                   vcf_path=vcf_file,
                   confidence_intervals=conf_intervals,
                   thread_no=thread_no)

    view_ob.parse_region(0, len(conf_intervals))


def create_output_dir_for_chromosome(output_dir, chr_name):
    """
    Create an internal directory inside the output directory to dump choromosomal summary files
    :param output_dir: Path to output directory
    :param chr_name: chromosome name
    :return: New directory path
    """
    path_to_dir = output_dir + chr_name + "/"
    if not os.path.exists(path_to_dir):
        os.mkdir(path_to_dir)

    summary_path = path_to_dir + "summary" + "/"
    if not os.path.exists(summary_path):
        os.mkdir(summary_path)

    candidate_dictionar_path = path_to_dir + "candidate_dictionaries" + "/"
    if not os.path.exists(candidate_dictionar_path):
        os.mkdir(candidate_dictionar_path)

    return path_to_dir


def chromosome_level_parallelization(chr_name, bam_file, ref_file, vcf_file, output_path, max_threads,
                                     confident_bed_tree, singleton_run=False):
    """
    This method takes one chromosome name as parameter and chunks that chromosome in max_threads.
    :param chr_name: Name of the chromosome
    :param bam_file: path to BAM file
    :param ref_file: path to reference FASTA file
    :param vcf_file: path to VCF file
    :param output_path: path to output directory
    :param max_threads: Maximum number of threads to run at one instance
    :param confident_bed_tree: tree containing confident bed intervals
    :param singleton_run: if running a chromosome independently
    :return:
    """
    chr_start_time = time.time()
    sys.stderr.write(TextColor.BLUE + "INFO: STARTING " + str(chr_name) + " PROCESSES" + "\n" + TextColor.END)
    # create dump directory inside output directory
    output_dir = create_output_dir_for_chromosome(output_path, chr_name)

    # entire length of chromosome
    c_intervals = confident_bed_tree[chr_name]
    total_segments = len(c_intervals)

    # .5MB segments at once
    each_chunk_size = 100

    if DEBUG_TEST_PARALLEL:
        each_chunk_size = 100
        total_segments = 700

    for i in tqdm(range(0, total_segments, each_chunk_size), file=sys.stdout, dynamic_ncols=True):
        start_position = i
        end_position = min(i + each_chunk_size, total_segments)
        chunk_intervals = c_intervals[start_position:end_position]

        # gather all parameters
        args = (chr_name, bam_file, ref_file, vcf_file, output_dir, chunk_intervals, i)
        p = multiprocessing.Process(target=parallel_run, args=args)
        p.start()

        # wait until we have room for new processes to start
        while True:
            if len(multiprocessing.active_children()) < max_threads:
                break

    if singleton_run:
        # wait for the last process to end before file processing
        while True:
            if len(multiprocessing.active_children()) == 0:
                break
        # remove summary files and make one file
        summary_file_to_csv(output_path, [chr_name])
        # merge_all_candidate_dictionaries(output_path, [chr_name])

        chr_end_time = time.time()
        sys.stderr.write(TextColor.RED + "CHROMOSOME PROCESSES FINISHED SUCCESSFULLY" + "\n")
        sys.stderr.write(
            TextColor.CYAN + "TOTAL TIME FOR GENERATING ALL RESULTS: " + str(chr_end_time - chr_start_time) + "\n")


def genome_level_parallelization(bam_file, ref_file, vcf_file, output_dir_path, max_threads, confident_bed_tree):
    """
    This method calls chromosome_level_parallelization for each chromosome.
    :param bam_file: path to BAM file
    :param ref_file: path to reference FASTA file
    :param vcf_file: path to VCF file
    :param output_dir_path: path to output directory
    :param max_threads: Maximum number of threads to run at one instance
    :param confident_bed_tree: tree containing confident bed intervals
    :return:
    """

    # --- NEED WORK HERE --- GET THE CHROMOSOME NAMES FROM THE BAM FILE
    chr_list = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11",
                "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19"]
    # chr_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]

    program_start_time = time.time()

    # chr_list = ["19"]

    # each chromosome in list
    for chr_name in chr_list:

        start_time = time.time()

        # do a chromosome level parallelization
        chromosome_level_parallelization(chr_name, bam_file, ref_file, vcf_file, output_dir_path,
                                         max_threads, confident_bed_tree)

        end_time = time.time()
        sys.stderr.write(TextColor.PURPLE + "FINISHED " + str(chr_name) + " PROCESSES" + "\n")
        sys.stderr.write(TextColor.CYAN + "TIME ELAPSED: " + str(end_time - start_time) + "\n")

    # wait for the last process to end before file processing
    while True:
        if len(multiprocessing.active_children()) == 0:
            break

    summary_file_to_csv(output_dir_path, chr_list)
    # merge_all_candidate_dictionaries(output_dir_path, chr_list)

    program_end_time = time.time()
    sys.stderr.write(TextColor.RED + "ALL PROCESSES FINISHED SUCCESSFULLY" + "\n")
    sys.stderr.write(TextColor.CYAN + "TOTAL TIME FOR GENERATING ALL RESULTS: " + str(program_end_time-program_start_time) + "\n")


def summary_file_to_csv(output_dir_path, chr_list):
    """
    Remove the abundant number of summary files and bind them to one
    :param output_dir_path: Path to the output directory
    :param chr_list: List of chromosomes
    :return:
    """
    for chr_name in chr_list:
        # here we dumped all the bed files
        path_to_dir = output_dir_path + chr_name + "/summary/"

        concatenated_file_name = output_dir_path + chr_name + ".csv"

        filemanager_object = FileManager()
        # get all bed file paths from the directory
        file_paths = filemanager_object.get_file_paths_from_directory(path_to_dir)
        # dump all bed files into one
        filemanager_object.concatenate_files(file_paths, concatenated_file_name)
        # delete all temporary files
        filemanager_object.delete_files(file_paths)
        # remove the directory
        os.rmdir(path_to_dir)


def merge_all_candidate_dictionaries(output_dir_path, chr_list):
    """
    Merge all dictionaries containing candidate alleles to one
    :param output_dir_path: Path to the output directory
    :param chr_list: List of chromosomes
    :return:
    """
    for chr_name in chr_list:
        # here we dumped all the bed files
        path_to_dir = output_dir_path + chr_name + "/candidate_dictionaries/"

        concatenated_file_name = output_dir_path + chr_name + "_candidate_alleles.pkl"

        filemanager_object = FileManager()
        # get all bed file paths from the directory
        file_paths = filemanager_object.get_file_paths_from_directory(path_to_dir)
        # dump all bed files into one
        filemanager_object.merge_dictionaries(file_paths, concatenated_file_name)
        # delete all temporary files
        filemanager_object.delete_files(file_paths)
        # remove the directory
        os.rmdir(path_to_dir)


def handle_output_directory(output_dir):
    """
    Process the output directory and return a valid directory where we save the output
    :param output_dir: Output directory path
    :return:
    """
    # process the output directory
    if output_dir[-1] != "/":
        output_dir += "/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # create an internal directory so we don't overwrite previous runs
    timestr = time.strftime("%m%d%Y_%H%M%S")
    internal_directory = "run_" + timestr + "/"
    output_dir = output_dir + internal_directory

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    return output_dir


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks.
    '''
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--bam",
        type=str,
        required=True,
        help="BAM file containing reads of interest."
    )
    parser.add_argument(
        "--ref",
        type=str,
        required=True,
        help="Reference corresponding to the BAM file."
    )
    parser.add_argument(
        "--vcf",
        type=str,
        required=True,
        help="VCF file path."
    )
    parser.add_argument(
        "--chromosome_name",
        type=str,
        help="Desired chromosome number E.g.: 3"
    )
    parser.add_argument(
        "--max_threads",
        type=int,
        default=8,
        help="Number of maximum threads for this region."
    )
    parser.add_argument(
        "--confident_bed",
        type=str,
        required=True,
        help="Path to confident BED file"
    )
    parser.add_argument(
        "--test",
        type=bool,
        default=False,
        help="If true then a dry test is run."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="candidate_finder_output/",
        help="Path to output directory."
    )

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.output_dir = handle_output_directory(FLAGS.output_dir)

    # if the confident bed is not empty then create the tree
    confident_interval_tree = build_chromosomal_interval_trees(FLAGS.confident_bed)

    if FLAGS.test is True:
        chromosome_output = create_output_dir_for_chromosome(FLAGS.output_dir, FLAGS.chromosome_name)
        view = View(chromosome_name=FLAGS.chromosome_name,
                    bam_file_path=FLAGS.bam,
                    reference_file_path=FLAGS.ref,
                    vcf_path=FLAGS.vcf,
                    output_file_path=chromosome_output,
                    confidence_intervals=confident_interval_tree[FLAGS.chromosome_name],
                    thread_no=1)
        test(view)
    elif FLAGS.chromosome_name is not None:
        chromosome_level_parallelization(FLAGS.chromosome_name, FLAGS.bam, FLAGS.ref, FLAGS.vcf, FLAGS.output_dir,
                                         FLAGS.max_threads, confident_interval_tree, singleton_run=True)
    else:
        genome_level_parallelization(FLAGS.bam, FLAGS.ref, FLAGS.vcf, FLAGS.output_dir,
                                     FLAGS.max_threads, confident_interval_tree)
