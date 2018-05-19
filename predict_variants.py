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
import operator
import pickle
from tqdm import tqdm
import os
from multiprocessing import Pool
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


def save_dictionary(dictionary, directory, file_name):
    """
    Save a dictionary to a file.
    :param dictionary: The dictionary to save
    :param directory: Directory to save the file to
    :param file_name: Name of file
    :return:
    """
    with open(directory + file_name, 'wb') as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)


prediction_dict = defaultdict(lambda: defaultdict(list))
reference_dict = defaultdict(lambda: defaultdict(tuple))


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
                            num_workers=num_workers,
                            pin_memory=gpu_mode
                            )

    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)
    ### FROM HERE
    '''
    # load the model
    if gpu_mode is False:
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']

        # In state dict keys there is an extra word inserted by model parallel: "module.". We remove it here
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if k[:7] == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v

        model = SeqResNet(image_channels=6, seq_length=20, num_classes=21)
        model.load_state_dict(new_state_dict)
        model.cpu()
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        model = SeqResNet(image_channels=6, seq_length=20, num_classes=21)
        model.load_state_dict(new_state_dict)
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()
    

    # Change model to 'eval' mode (BN uses moving mean/var).
    model.eval()'''
    # TO HERE

    for images, labels, positional_info in tqdm(testloader, file=sys.stdout, dynamic_ncols=True):
        ### FROM HERE
        '''
        images = Variable(images, volatile=True)

        if gpu_mode:
            images = images.cuda()

        output_preds = model(images)
        # One dimensional softmax is used to convert the logits to probability distribution
        m = nn.Softmax(dim=2)
        soft_probs = m(output_preds)
        output_preds = soft_probs.cpu()'''

        # record each of the predictions from a batch prediction
        batches = labels.size(0)
        seqs = labels.size(1)
        chr_name, start_positions, reference_seqs, allele_dict_paths = positional_info

        for batch in range(batches):
            allele_dict_path = allele_dict_paths[batch]
            chromosome_name = chr_name[batch]
            reference_seq = reference_seqs[batch]
            current_genomic_position = int(start_positions[batch])
            insert_index = 0
            for seq in range(FLANK_SIZE, seqs-FLANK_SIZE+1):

                ref_base = reference_seq[seq]

                '''
                preds = output_preds[batch, seq, :].data
                top_n, top_i = preds.topk(1)
                predicted_label = top_i[0]
                '''

                true_label = labels[batch, seq]
                fake_probs = [0.0] * 21
                fake_probs[true_label] = 1.0
                reference_dict[current_genomic_position][insert_index] = (ref_base, allele_dict_path)
                prediction_dict[current_genomic_position][insert_index].append((true_label, fake_probs))
                if reference_seq[seq+1] != '*':
                    insert_index = 0
                    current_genomic_position += 1
                else:
                    insert_index += 1

    return chromosome_name


def produce_vcf(chromosome_name, bam_file_path, sample_name, output_dir):
    """
    Convert prediction dictionary to a VCF file
    :param chromosome_name: Chromosome name
    :param bam_file_path: Path to the BAM file
    :param sample_name: Name of the sample in the BAM file
    :param output_dir: Output directory
    :return:
    """
    # object that can write and handle VCF
    vcf_writer = VCFWriter(bam_file_path, sample_name, output_dir)

    # collate multi-allelic records to a single record
    all_calls = []
    for pos in prediction_dict.keys():
        if len(prediction_dict[pos]) == 1:
            ref, alts, genotype, qual, gq = \
                vcf_writer.process_snp_or_del(pos, prediction_dict[pos], reference_dict[pos])
            if genotype != 0 and '.' not in alts:
                all_calls.append((chromosome_name, int(pos), int(pos + 1), ref, alts, genotype, qual, gq))
        # else:
        #     print('IN: ', pos)

    # sort based on position
    all_calls.sort(key=operator.itemgetter(1))
    last_end = 0
    for record in all_calls:
        # get the record filter ('PASS' or not)
        rec_filter = vcf_writer.get_filter(record, last_end)
        # get proper alleles. INDEL alleles are handled here.
        record = vcf_writer.get_proper_alleles(record)
        chrm, st_pos, end_pos, ref, alt_field, genotype, phred_qual, phred_gq = record
        # if genotype is not HOM keep track of where the previous record ended
        if genotype != '0/0':
            # HOM
            last_end = end_pos
        # add the record to VCF
        vcf_writer.write_vcf_record(chrm, st_pos, end_pos, ref, alt_field, genotype, phred_qual, phred_gq, rec_filter)


def produce_vcf_records(chromosome_name, pos_list):
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


def call_variant(csv_file, batch_size, model_path, gpu_mode, num_workers, bam_file_path, sample_name, output_dir,
                 max_threads):
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "SAMPLE NAME: " + sample_name + "\n")
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "PLEASE USE --sample_name TO CHANGE SAMPLE NAME.\n")
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "OUTPUT DIRECTORY: " + output_dir + "\n")
    chr_name = predict(csv_file, batch_size, model_path, gpu_mode, num_workers)
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "PREDICTION COMPLETED SUCCESSFULLY.\n")
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "GENERATING VCF.\n")
    genomic_positions = list(prediction_dict.keys())
    total_segments = len(genomic_positions)
    each_chunk_size = int(total_segments/max_threads)
    arg_list = []
    for i in tqdm(range(0, total_segments, each_chunk_size), file=sys.stdout, dynamic_ncols=True):
        start_position = i
        end_position = min(i + each_chunk_size, total_segments)
        pos_list = genomic_positions[start_position:end_position]

        # gather all parameters
        args = (chr_name, pos_list)
        arg_list.append(args)

    pool = Pool(processes=max_threads)
    results = [list() for i in range(len(arg_list))]
    for idx in range(len(arg_list)):
        results[idx] = pool.apply_async(produce_vcf_records, arg_list[idx])

    vcf_ready_calls = list()
    for idx in range(len(arg_list)):
        vcf_ready_calls.append(results[idx].get())

    vcf_ready_calls = [item for list_of_calls in vcf_ready_calls for item in list_of_calls]

    write_vcf(bam_file_path, sample_name, output_dir, vcf_ready_calls)
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "VARIANT CALLING COMPLETE.\n")



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
        default=4,
        help="Number of maximum threads for this region."
    )
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.output_dir = handle_output_directory(FLAGS.output_dir)
    call_variant(FLAGS.csv_file,
                 FLAGS.batch_size,
                 FLAGS.model_path,
                 FLAGS.gpu_mode,
                 FLAGS.num_workers,
                 FLAGS.bam_file,
                 FLAGS.sample_name,
                 FLAGS.output_dir,
                 FLAGS.max_threads)

