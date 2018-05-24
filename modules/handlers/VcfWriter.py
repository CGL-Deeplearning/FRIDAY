from pysam import VariantFile, VariantHeader
from collections import defaultdict
from modules.handlers.BamHandler import BamHandler
import math
import time
import numpy as np

DEL_TYPE = '3'
IN_TYPE = '2'
MATCH_TYPE = '1'
HOM = 0
HET = 1
HOM_ALT = 2


class VCFWriter:
    def __init__(self, bam_file_path, sample_name, output_dir):
        self.bam_handler = BamHandler(bam_file_path)
        bam_file_name = bam_file_path.rstrip().split('/')[-1].split('.')[0]
        vcf_header = self.get_vcf_header(sample_name)
        time_str = time.strftime("%m%d%Y_%H%M%S")

        self.vcf_file = VariantFile(output_dir + bam_file_name + '_' + time_str + '.vcf', 'w', header=vcf_header)

    def write_vcf_record(self, chrm, st_pos, end_pos, ref, alts, genotype, qual, gq, rec_filter):
        alleles = tuple([ref]) + tuple(alts)
        genotype = self.get_genotype_tuple(genotype)
        end_pos = int(end_pos) + 1
        st_pos = int(st_pos)
        vcf_record = self.vcf_file.new_record(contig=str(chrm), start=st_pos, stop=end_pos, id='.', qual=qual,
                                              filter=rec_filter, alleles=alleles, GT=genotype, GQ=gq)
        self.vcf_file.write(vcf_record)

    @staticmethod
    def prediction_label_to_allele(label):
        label_to_allele = {0:  ['0', '0'],  1:  ['0', '1'], 2:  ['1', '1'], 3:  ['0', '2'], 4:  ['2', '2'],
                           5:  ['1', '2']}
        return label_to_allele[label]

    @staticmethod
    def get_qual_and_gq(probabilities, predicted_class):
        qual = 1.0 - probabilities[0]
        phred_qual = min(60, -10 * np.log10(1 - qual) if 1 - qual >= 0.0000001 else 60)
        phred_qual = math.ceil(phred_qual * 100.0) / 100.0

        gq = probabilities[predicted_class]
        phred_gq = min(60, -10 * np.log10(1 - gq) if 1 - gq >= 0.0000001 else 60)
        phred_gq = math.ceil(phred_gq * 100.0) / 100.0
        return phred_qual, phred_gq

    @staticmethod
    def solve_multiple_alts(alts, ref):
        type1, type2 = alts[0][1], alts[1][1]
        alt1, alt2 = alts[0][0], alts[1][0]
        if type1 == DEL_TYPE and type2 == DEL_TYPE:
            if len(alt2) > len(alt1):
                return alt2, ref, alt2[0] + alt2[len(alt1):]
            else:
                return alt1, ref, alt1[0] + alt1[len(alt2):]
        elif type1 == IN_TYPE and type2 == IN_TYPE:
            return ref, alt1, alt2
        elif type1 == DEL_TYPE or type2 == DEL_TYPE:
            if type1 == DEL_TYPE and type2 == IN_TYPE:
                return alt1, ref, alt1 + alt2[1:]
            elif type1 == IN_TYPE and type2 == DEL_TYPE:
                return alt2, alt2 + alt1[1:], ref
            elif type1 == DEL_TYPE:
                return alt1, ref, alt2
            elif type2 == DEL_TYPE:
                return alt2, alt1, ref
        else:
            return ref, alt1, alt2

    @staticmethod
    def solve_single_alt(alts, ref):
        # print(alts)
        alt1, alt_type = alts
        if alt_type == DEL_TYPE:
            return alt1, ref, '.'
        return ref, alt1, '.'

    @staticmethod
    def get_genotype_tuple(genotype):
        split_values = genotype.split('/')
        split_values = [int(x) for x in split_values]
        return tuple(split_values)

    @staticmethod
    def get_genotype_for_single_allele(records):
        for record in records:
            probs = [record[8], record[9], record[10]]
            genotype_list = ['0/0', '0/1', '1/1']
            gq, index = max([(v, i) for i, v in enumerate(probs)])
            qual = sum(probs) - probs[0]
            ref = record[3]
            alt_with_types = list()
            alt_with_types.append((record[4], record[6]))
            # print(alt_with_types)
            ref, alt1, alt2 = VCFWriter.solve_single_alt(alt_with_types[0], ref)
            # print(ref, rec_alt1, rec_alt2)
            phred_qual = min(60, -10 * np.log10(1 - qual) if 1 - qual >= 0.0000001 else 60)
            phred_qual = math.ceil(phred_qual * 100.0) / 100.0
            phred_gq = min(60, -10 * np.log10(1 - gq) if 1 - gq >= 0.0000001 else 60)
            phred_gq = math.ceil(phred_gq * 100.0) / 100.0

            return record[0], record[1], record[2], ref, [alt1, alt2], genotype_list[index], phred_qual, phred_gq

    @staticmethod
    def process_prediction(pos, predictions):
        # get the list of prediction labels
        list_prediction_labels = [label for label, probs in predictions[0]]
        predicted_class = max(set(list_prediction_labels), key=list_prediction_labels.count)

        # get alts from label
        genotype = VCFWriter.prediction_label_to_allele(predicted_class)

        # get the probabilities
        list_prediction_probabilities = [probs for label, probs in predictions[0]]
        num_classes = len(list_prediction_probabilities[0])
        min_probs_for_each_class = [min(l[i] for l in list_prediction_probabilities) for i in range(num_classes)]

        # normalize the probabilities
        sum_of_probs = sum(min_probs_for_each_class) if sum(min_probs_for_each_class) > 0 else 1
        if sum(min_probs_for_each_class) <= 0:
            print("SUM ZERO ENCOUNTERED IN: ", pos, predictions)
            exit()
        probabilities = [float(i) / sum_of_probs for i in min_probs_for_each_class]

        qual, gq = VCFWriter.get_qual_and_gq(probabilities, predicted_class)

        return genotype, qual, gq

    @staticmethod
    def get_proper_alleles(record):
        ref, alt_field, genotype, phred_qual, phred_gq = record
        if len(alt_field) == 1:
            ref, alt1, alt2 = VCFWriter.solve_single_alt(alt_field[0], ref)
        else:
            ref, alt1, alt2 = VCFWriter.solve_multiple_alts(alt_field, ref)

        gts = genotype
        refined_alt = []
        if gts[0] == '1' or gts[1] == '1':
            refined_alt.append(alt_field[0][0])
        if gts[0] == '2' or gts[1] == '2':
            refined_alt.append(alt_field[1][0])
        if gts[0] == '0' and gts[1] == '0':
            refined_alt.append('.')

        record = ref, refined_alt, phred_qual, phred_gq, genotype

        return record

    @staticmethod
    def get_filter(record, last_end):
        chrm, st_pos, end_pos, ref, alt_field, genotype, phred_qual, phred_gq = record
        if st_pos < last_end:
            return 'conflictPos'
        if genotype == '0/0':
            return 'refCall'
        if phred_qual < 0:
            return 'lowQUAL'
        if phred_gq < 0:
            return 'lowGQ'
        return 'PASS'

    def get_vcf_header(self, sample_name):
        header = VariantHeader()
        items = [('ID', "PASS"),
                 ('Description', "All filters passed")]
        header.add_meta(key='FILTER', items=items)
        items = [('ID', "refCall"),
                 ('Description', "Call is homozygous")]
        header.add_meta(key='FILTER', items=items)
        items = [('ID', "lowGQ"),
                 ('Description', "Low genotype quality")]
        header.add_meta(key='FILTER', items=items)
        items = [('ID', "lowQUAL"),
                 ('Description', "Low variant call quality")]
        header.add_meta(key='FILTER', items=items)
        items = [('ID', "conflictPos"),
                 ('Description', "Overlapping record")]
        header.add_meta(key='FILTER', items=items)
        items = [('ID', "GT"),
                 ('Number', 1),
                 ('Type', 'String'),
                 ('Description', "Genotype")]
        header.add_meta(key='FORMAT', items=items)
        items = [('ID', "GQ"),
                 ('Number', 1),
                 ('Type', 'Float'),
                 ('Description', "Genotype Quality")]
        header.add_meta(key='FORMAT', items=items)
        bam_sqs = self.bam_handler.get_header_sq()
        for sq in bam_sqs:
            id = sq['SN']
            ln = sq['LN']
            items = [('ID', id),
                     ('length', ln)]
            header.add_meta(key='contig', items=items)

        header.add_sample(sample_name)

        return header
