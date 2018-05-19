import argparse
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from modules.models.Seq2Seq_CNN import SeqResNet
from modules.core.dataloader import SequenceDataset
from modules.handlers.TextColor import TextColor
from collections import defaultdict
from modules.handlers.VcfWriter import VCFWriter
from multiprocessing import Pool
import operator
import pickle
from tqdm import tqdm
import os
"""
This script uses a trained model to call variants on a given set of images generated from the genome.
The process is:
- Create a prediction table/dictionary using a trained neural network
- Convert those predictions to a VCF file

INPUT:
- A trained model
- Set of images for prediction

Output:
- A VCF file containing all the variants.
"""
FLANK_SIZE = 5
WINDOW_SIZE = 300


def load_dictionary(path_to_dictionary):
    """
    Load a dictionary
    :param path_to_dictionary: Path to the location where dictionary is saved
    :return:
    """
    return pickle.load(open(path_to_dictionary, 'rb'))


def prediction_dictionary_structure():
    return defaultdict(list)


def reference_dictionary_structure():
    return defaultdict(int)


def produce_vcf_records(arg_tuple):
    """
    Convert prediction dictionary to a VCF file
    :param: arg_tuple: Tuple of arguments containing these values:
    - chromosome_name: Chromosome name
    - pos_list: List of positions where we will search for variants
    - prediction_dict: prediction dictionary containing predictions of each image records
    - reference_dict: Dictionary containing reference information
    - bam_file_path: Path to the BAM file
    - sample_name: Name of the sample in the BAM file
    - output_dir: Output directory
    - thread_id: Unique id assigned to each thread
    :return:
    """
    # object that can write and handle VCF
    # vcf_writer = VCFWriter(bam_file_path, sample_name, output_dir, thread_id)
    chromosome_name, pos_list, prediction_dict, reference_dict = arg_tuple
    # collate multi-allelic records to a single record
    all_calls = []
    for pos in pos_list:
        # if len(prediction_dict[pos]) == 1:
        ref, alts, genotype, qual, gq = \
            VCFWriter.process_snp_or_del(pos, prediction_dict[pos], reference_dict[pos])
        if genotype != 0 and '.' not in alts:
            all_calls.append((chromosome_name, int(pos), int(pos + 1), ref, alts, genotype, qual, gq))
        # else:
        #     print('IN: ', pos)

    # sort based on position
    all_calls.sort(key=operator.itemgetter(1))
    last_end = 0
    vcf_ready_records = []

    for record in all_calls:
        # get the record filter ('PASS' or not)
        rec_filter = VCFWriter.get_filter(record, last_end)
        # get proper alleles. INDEL alleles are handled here.
        record = VCFWriter.get_proper_alleles(record)
        chrm, st_pos, end_pos, ref, alt_field, genotype, phred_qual, phred_gq = record
        # if genotype is not HOM keep track of where the previous record ended
        if genotype != '0/0':
            # HOM
            last_end = end_pos
        # add the record to VCF
        vcf_ready_records.append((chrm, st_pos, end_pos, ref, alt_field, genotype, phred_qual, phred_gq, rec_filter))

    return vcf_ready_records


def write_vcf(bam_file_path, sample_name, output_dir, vcf_calls):
    vcf_writer = VCFWriter(bam_file_path, sample_name, output_dir)
    for record in vcf_calls:
        chrm, st_pos, end_pos, ref, alts, genotype, qual, gq, rec_filter = record
        vcf_writer.write_vcf_record(chrm, st_pos, end_pos, ref, alts, genotype, qual, gq, rec_filter)


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

    return output_dir


def call_variant(chr_name, allele_dict_path, pred_dict_path, ref_dict_path, bam_file_path, sample_name, output_dir,
                 max_threads):
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "SAMPLE NAME: " + sample_name + "\n")
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "PLEASE USE --sample_name TO CHANGE SAMPLE NAME.\n")
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "OUTPUT DIRECTORY: " + output_dir + "\n")
    allele_dict = load_dictionary(allele_dict_path)
    prediction_dict = load_dictionary(pred_dict_path)
    reference_dict = load_dictionary(ref_dict_path)
    intersected_positions = sorted(list(set(list(prediction_dict.keys())) & set(list(allele_dict.keys()))))

    total_segments = len(intersected_positions)
    each_chunk_size = int(total_segments/max_threads)
    arg_list = []
    for i in tqdm(range(0, total_segments, each_chunk_size), file=sys.stdout, dynamic_ncols=True):
        start_position = i
        end_position = min(i + each_chunk_size, total_segments)
        pos_list = intersected_positions[start_position:end_position]

        # gather all parameters
        args = (chr_name, pos_list, prediction_dict, reference_dict)
        arg_list.append(args)

    pool = Pool(processes=len(arg_list))

    vcf_ready_calls = pool.map(produce_vcf_records, arg_list)
    vcf_ready_calls = [item for list_of_calls in vcf_ready_calls for item in list_of_calls]

    write_vcf(bam_file_path, sample_name, output_dir, vcf_ready_calls)


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chromosome_name",
        type=str,
        required=True,
        help="Chromosome name."
    )
    parser.add_argument(
        "--allele_dictionary",
        type=str,
        required=True,
        help="Path to allele dictionary that contains alleles."
    )
    parser.add_argument(
        "--prediction_dictionary",
        type=str,
        required=True,
        help="Path to allele dictionary that contains predictions."
    )
    parser.add_argument(
        "--reference_dictionary",
        type=str,
        required=True,
        help="Path to allele dictionary that contains predictions."
    )
    parser.add_argument(
        "--bam_file",
        type=str,
        required=True,
        help="Path to the BAM file."
    )
    parser.add_argument(
        "--sample_name",
        type=str,
        required=False,
        default='NA12878',
        help="Sample name of the sequence."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default='vcf_output',
        help="Output directory."
    )
    parser.add_argument(
        "--max_threads",
        type=int,
        default=4,
        help="Number of maximum threads for this region."
    )
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.output_dir = handle_output_directory(FLAGS.output_dir)
    call_variant(FLAGS.chromosome_name,
                 FLAGS.allele_dictionary,
                 FLAGS.prediction_dictionary,
                 FLAGS.reference_dictionary,
                 FLAGS.bam_file,
                 FLAGS.sample_name,
                 FLAGS.output_dir,
                 FLAGS.max_threads)

