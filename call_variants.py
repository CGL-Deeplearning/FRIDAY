import argparse
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import multiprocessing
from modules.core.dataloader_test import SequenceDataset
from modules.handlers.TextColor import TextColor
from collections import defaultdict
from modules.handlers.VcfWriter import VCFWriter
from modules.handlers.FileManager import FileManager
from modules.models.ModelHandler import ModelHandler
import operator
import pickle
from tqdm import tqdm
import os
import time
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
SNP = 1
IN = 2
DEL = 3
HOM = 0
HET = 1
HOM_ALT = 2

prediction_dict = defaultdict(list)
reference_dict = defaultdict(tuple)


def predict(test_file, batch_size, model_path, gpu_mode, num_workers):
    """
    Create a prediction table/dictionary of an images set using a trained model.
    :param test_file: File to predict on
    :param batch_size: Batch size used for prediction
    :param model_path: Path to a trained model
    :param gpu_mode: If true, predictions will be done over GPU
    :param num_workers: Number of workers to be used by the dataloader
    :return: Prediction dictionary
    """
    # the prediction table/dictionary
    chromosome_name = ''
    transformations = transforms.Compose([transforms.ToTensor()])

    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)

    test_dset = SequenceDataset(test_file, transformations)
    testloader = DataLoader(test_dset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers
                            )

    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    # load the model
    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)

    if os.path.isfile(model_path) is False:
        sys.stderr.write(TextColor.RED + "ERROR: INVALID PATH TO MODEL\n")
        exit(1)

    sys.stderr.write(TextColor.GREEN + "INFO: MODEL LOADING\n" + TextColor.END)
    encoder_model, decoder_model, hidden_size, gru_layers, epochs = ModelHandler.load_model_for_training(model_path,
                                                                                                         input_channels=10,
                                                                                                         num_classes=6)

    sys.stderr.write(TextColor.GREEN + "INFO: MODEL LOADED\n" + TextColor.END)

    if gpu_mode:
        encoder_model = torch.nn.DataParallel(encoder_model).cuda()
        decoder_model = torch.nn.DataParallel(decoder_model).cuda()

    encoder_model.eval()
    decoder_model.eval()

    sys.stderr.write(TextColor.PURPLE + 'MODEL LOADED\n' + TextColor.END)

    with torch.no_grad():
        for images, labels, positional_info in tqdm(testloader, ncols=50):
            if gpu_mode:
                images = images.cuda()
                labels = labels.cuda()

            encoder_hidden = torch.FloatTensor(labels.size(0), gru_layers * 2, hidden_size).zero_()

            if gpu_mode:
                encoder_hidden = encoder_hidden.cuda()

            chr_name, start_positions, reference_seqs, allele_dict_paths = positional_info

            unrolling_genomic_position = np.zeros((images.size(0)), dtype=np.int64)

            context_vector, hidden_encoder = encoder_model(images, encoder_hidden)
            current_batch_size = images.size(0)
            seq_length = images.size(2)
            for seq_index in range(0, seq_length):
                attention_index = torch.from_numpy(np.asarray([seq_index] * current_batch_size)).view(-1, 1)

                attention_index_onehot = torch.FloatTensor(current_batch_size, seq_length)

                attention_index_onehot.zero_()
                attention_index_onehot.scatter_(1, attention_index, 1)

                output_dec, decoder_hidden, attn = decoder_model(attention_index_onehot,
                                                                 context_vector=context_vector,
                                                                 encoder_hidden=hidden_encoder)

                # One dimensional softmax is used to convert the logits to probability distribution
                m = nn.Softmax(dim=1)
                soft_probs = m(output_dec)
                output_preds = soft_probs.cpu()

                # record each of the predictions from a batch prediction
                for batch in range(current_batch_size):
                    allele_dict_path = allele_dict_paths[batch]
                    chromosome_name = chr_name[batch]
                    reference_seq = reference_seqs[batch]
                    # current_genomic_position = int(start_positions[batch])
                    current_genomic_position = int(start_positions[batch]) + unrolling_genomic_position[batch]

                    ref_base = reference_seq[seq_index]

                    if ref_base == '*':
                        continue

                    true_label = labels[batch, seq_index - index_start]
                    fake_probs = [0.0] * 6
                    fake_probs[true_label] = 1.0
                    top_n, top_i = torch.FloatTensor(fake_probs).topk(1)
                    predicted_label = top_i[0].item()
                    reference_dict[current_genomic_position] = (ref_base, allele_dict_path)
                    prediction_dict[current_genomic_position].append((predicted_label, fake_probs))

                    # preds = output_preds[batch, :].data
                    # top_n, top_i = preds.topk(1)
                    # predicted_label = top_i[0].item()
                    # reference_dict[current_genomic_position] = (ref_base, allele_dict_path)
                    # prediction_dict[current_genomic_position].append((predicted_label, preds))

                    if ref_base != '*':
                        unrolling_genomic_position[batch] += 1
    return chromosome_name


def get_record_from_prediction(pos, alleles):
    predictions = prediction_dict[pos]
    genotype, qual, gq = VCFWriter.process_prediction(pos, predictions)
    alts = list(allele[0] for allele in alleles)
    ref_base = reference_dict[pos][0][0]
    return ref_base, alts, genotype, qual, gq


def produce_vcf_records(chromosome_name, output_dir, thread_no, pos_list):
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
    # collate multi-allelic records to a single record
    current_allele_dict = ''
    allele_dict = {}
    record_file = open(output_dir + chromosome_name + "_" + str(thread_no) + ".tsv", 'w')
    for pos in pos_list:
        allele_dict_path = reference_dict[pos][1]
        if allele_dict_path != current_allele_dict:
            allele_dict = pickle.load(open(allele_dict_path, 'rb'))
            current_allele_dict = allele_dict_path

        if pos not in allele_dict:
            continue
        alleles = allele_dict[pos]

        record = get_record_from_prediction(pos, alleles)

        if record is None:
            continue

        ref_base, alts, genotype, qual, gq = record

        if genotype == '0/0':
            continue
        # print('BEFORE', record)
        record = VCFWriter.get_proper_alleles(record)
        ref, alts, qual, gq, genotype = record
        # print('AFTER', record)
        if len(alts) == 1:
            alts.append('.')
        rec_end = int(pos + len(ref) - 1)
        record_string = chromosome_name + "\t" + str(pos) + "\t" + str(rec_end) + "\t" + ref + "\t" + '\t'.join(alts) \
                        + "\t" + genotype + "\t" + str(qual) + "\t" + str(gq) + "\t" + "\n"
        record_file.write(record_string)


def merge_call_files(vcf_file_directory):
    filemanager_object = FileManager()
    # get all bed file paths from the directory
    file_paths = filemanager_object.get_file_paths_from_directory(vcf_file_directory)
    all_records = []
    for file_path in file_paths:
        with open(file_path, 'r') as tsv:
            for line in tsv:
                chr_name, pos_st, pos_end, ref, alt1, alt2, genotype, qual, gq = line.strip().split('\t')
                alts = []
                pos_st, pos_end, qual, gq = int(pos_st), int(pos_end), float(qual), float(gq)
                if alt1 != '.':
                    alts.append(alt1)
                if alt2 != '.':
                    alts.append(alt2)
                all_records.append((chr_name, pos_st, pos_end, ref, alts, genotype, qual, gq))

    filemanager_object.delete_files(file_paths)
    os.rmdir(vcf_file_directory)

    return all_records


def call_variant(csv_file, batch_size, model_path, gpu_mode, num_workers, bam_file_path, sample_name, output_dir,
                 vcf_dir, max_threads):
    program_start_time = time.time()
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "SAMPLE NAME: " + sample_name + "\n")
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "PLEASE USE --sample_name TO CHANGE SAMPLE NAME.\n")
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "OUTPUT DIRECTORY: " + output_dir + "\n")
    chr_name = predict(csv_file, batch_size, model_path, gpu_mode, num_workers)
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "PREDICTION GENERATED SUCCESSFULLY.\n")
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "COMPILING PREDICTIONS TO CALL VARIANTS.\n")

    pos_list = list(prediction_dict.keys())
    each_chunk_size = int(len(pos_list) / max_threads)
    thread_no = 1
    # produce_vcf_records(chr_name, vcf_dir, thread_no, pos_list)
    # exit()

    for i in tqdm(range(0, len(pos_list), each_chunk_size), file=sys.stdout, dynamic_ncols=True):
        start_position = i
        end_position = min(i + each_chunk_size, len(pos_list))

        sub_pos = pos_list[start_position:end_position]
        # gather all parameters
        args = (chr_name, vcf_dir, thread_no, sub_pos)
        p = multiprocessing.Process(target=produce_vcf_records, args=args)
        p.start()
        thread_no += 1

        # wait until we have room for new processes to start
        while True:
            if len(multiprocessing.active_children()) < max_threads:
                break

    # wait until we have room for new processes to start
    while True:
        if len(multiprocessing.active_children()) == 0:
            break
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "VARIANT CALLING COMPLETE.\n")
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "MERGING FILES.\n")
    all_calls = merge_call_files(vcf_dir)

    # sort based on position
    all_calls.sort(key=operator.itemgetter(1))
    # print(all_calls)
    last_end = 0
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "WRITING VCF.\n")
    vcf_writer = VCFWriter(bam_file_path, sample_name, output_dir)
    for record in all_calls:
        # get the record filter ('PASS' or not)
        rec_filter = VCFWriter.get_filter(record, last_end)
        # get proper alleles. INDEL alleles are handled here.
        # record = VCFWriter.get_proper_alleles(record)
        chrm, st_pos, end_pos, ref, alt_field, genotype, phred_qual, phred_gq = record
        # if genotype is not HOM keep track of where the previous record ended
        if genotype != '0/0':
            # HOM
            last_end = end_pos
        # add the record to VCF
        vcf_writer.write_vcf_record(chrm, st_pos, end_pos, ref, alt_field, genotype, phred_qual, phred_gq, rec_filter)

    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "VARIANT CALLING COMPLETE.\n")
    program_end_time = time.time()

    sys.stderr.write(TextColor.PURPLE + "TIME ELAPSED: " + str(program_end_time - program_start_time) + "\n")


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

    vcf_path = output_dir + "vcfs" + "/"
    if not os.path.exists(vcf_path):
        os.mkdir(vcf_path)

    return output_dir, vcf_path


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="CSV file containing all image segments for prediction."
    )
    parser.add_argument(
        "--bam_file",
        type=str,
        required=True,
        help="Path to the BAM file."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=100,
        help="Batch size for testing, default is 100."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=4,
        help="Batch size for testing, default is 100."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='./CNN.pkl',
        help="Saved model path."
    )
    parser.add_argument(
        "--gpu_mode",
        type=bool,
        default=False,
        help="If true then cuda is on."
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
        default=8,
        help="Number of maximum threads for this region."
    )
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.output_dir, vcf_dir = handle_output_directory(FLAGS.output_dir)
    call_variant(FLAGS.csv_file,
                 FLAGS.batch_size,
                 FLAGS.model_path,
                 FLAGS.gpu_mode,
                 FLAGS.num_workers,
                 FLAGS.bam_file,
                 FLAGS.sample_name,
                 FLAGS.output_dir,
                 vcf_dir,
                 FLAGS.max_threads)

