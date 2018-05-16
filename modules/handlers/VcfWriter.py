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
        label_to_allele = {0:  ['*', '*'],  1:  ['*', '.'], 2:  ['*', 'A'], 3:  ['*', 'C'], 4:  ['*', 'G'],
                           5:  ['*', 'T'],  6:  ['.', '.'], 7:  ['.', 'A'], 8:  ['.', 'C'], 9:  ['.', 'G'],
                           10: ['.', 'T'],  11: ['A', 'A'], 12: ['A', 'C'], 13: ['A', 'G'], 14: ['A', 'T'],
                           15: ['C', 'C'],  16: ['C', 'G'], 17: ['C', 'T'], 18: ['G', 'G'], 19: ['G', 'T'],
                           20: ['T', 'T']}
        return label_to_allele[label]

    @staticmethod
    def get_reference_class(reference_base):
        base_to_letter_map = {'**': 0, '*.': 1, '*A': 2, '*C': 3, '*G': 4, '*T': 5,
                              '..': 6, '.A': 7, '.C': 8, '.G': 9, '.T': 10,
                              'AA': 11, 'AC': 12, 'AG': 13, 'AT': 14,
                              'CC': 15, 'CG': 16, 'CT': 17,
                              'GG': 18, 'GT': 19,
                              'TT': 20}
        if reference_base == 'N':
            reference_base = '*'
        reference_allele = reference_base + reference_base
        return base_to_letter_map[reference_allele]

    @staticmethod
    def get_qual_and_gq(probabilities, predicted_class, reference_base):
        qual = 1.0 - probabilities[VCFWriter.get_reference_class(reference_base)]
        phred_qual = min(60, -10 * np.log10(1 - qual) if 1 - qual >= 0.0000001 else 60)
        phred_qual = math.ceil(phred_qual * 100.0) / 100.0

        gq = probabilities[predicted_class]
        phred_gq = min(60, -10 * np.log10(1 - gq) if 1 - gq >= 0.0000001 else 60)
        phred_gq = math.ceil(phred_gq * 100.0) / 100.0
        return phred_qual, phred_gq

    @staticmethod
    def process_snp_or_del(predictions, reference):
        # get the list of prediction labels
        list_prediction_labels = [label for label, probs in predictions[0]]
        predicted_class = max(set(list_prediction_labels), key=list_prediction_labels.count)

        # get alts from label
        alts = VCFWriter.prediction_label_to_allele(predicted_class)
        reference_base = reference[0]

        # get the probabilities
        list_prediction_probabilities = [probs for label, probs in predictions[0]]
        num_classes = len(list_prediction_probabilities[0])
        min_probs_for_each_class = [min(l[i] for l in list_prediction_probabilities) for i in range(num_classes)]

        # normalize the probabilities
        probabilities = [float(i) / sum(min_probs_for_each_class) for i in min_probs_for_each_class]
        if alts[0] == reference_base and alts[1] == reference_base:
            genotype = HOM
            qual, gq = VCFWriter.get_qual_and_gq(probabilities, predicted_class, reference_base)
            return reference_base, alts, genotype, qual, gq
        elif alts[0] != reference_base and alts[1] != reference_base:
            if alts[0] == alts[1]:
                alts = [alts[0]]
            genotype = HOM_ALT
            qual, gq = VCFWriter.get_qual_and_gq(probabilities, predicted_class, reference_base)
            return reference_base, alts, genotype, qual, gq
        else:
            alts = [alts[0]] if alts[0] != reference_base else [alts[1]]
            genotype = HOM_ALT
            qual, gq = VCFWriter.get_qual_and_gq(probabilities, predicted_class, reference_base)
            return reference_base, alts, genotype, qual, gq

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
    def get_genotype_for_multiple_allele(records):

        ref = '.'
        st_pos = 0
        end_pos = 0
        chrm = ''
        rec_alt1 = '.'
        rec_alt2 = '.'
        alt_probs = defaultdict(list)
        alt_with_types = []
        for record in records:
            chrm = record[0]
            st_pos = record[1]
            end_pos = record[2]
            ref = record[3]
            alt1 = record[4]
            alt2 = record[5]
            if alt1 != '.' and alt2 != '.':
                rec_alt1 = alt1
                rec_alt2 = alt2
                alt_probs['both'] = (record[8:])
            else:
                alt_probs[alt1] = (record[8:])
                alt_with_types.append((alt1, record[6]))

        p00 = min(alt_probs[rec_alt1][0], alt_probs[rec_alt2][0], alt_probs['both'][0])
        p01 = min(alt_probs[rec_alt1][1], alt_probs['both'][1])
        p11 = min(alt_probs[rec_alt1][2], alt_probs['both'][2])
        p02 = min(alt_probs[rec_alt2][1], alt_probs['both'][1])
        p22 = min(alt_probs[rec_alt2][2], alt_probs['both'][2])
        p12 = min(max(alt_probs[rec_alt1][1], alt_probs[rec_alt1][2]),
                  max(alt_probs[rec_alt2][1], alt_probs[rec_alt2][2]),
                  max(alt_probs['both'][1], alt_probs['both'][2]))
        # print(alt_probs)
        prob_list = [p00, p01, p11, p02, p22, p12]
        # print(prob_list)
        sum_probs = sum(prob_list)
        # print(sum_probs)
        normalized_list = [(float(i) / sum_probs) if sum_probs else 0 for i in prob_list]
        prob_list = normalized_list
        # print(prob_list)
        # print(sum(prob_list))
        genotype_list = ['0/0', '0/1', '1/1', '0/2', '2/2', '1/2']
        gq, index = 0, 0
        for i, prob in enumerate(prob_list):
            if gq <= prob and prob > 0:
                index = i
                gq = prob
        qual = sum(prob_list) - prob_list[0]
        if index == 5:
            ref, rec_alt1, rec_alt2 = VCFWriter.solve_multiple_alts(alt_with_types, ref)
        else:
            if index <= 2:
                ref, rec_alt1, rec_alt2 = VCFWriter.solve_single_alt(alt_with_types[0], ref)
            else:
                ref, rec_alt2, rec_alt1 = VCFWriter.solve_single_alt(alt_with_types[1], ref)

        phred_qual = min(60, -10 * np.log10(1 - qual) if 1 - qual >= 0.0000001 else 60)
        phred_qual = math.ceil(phred_qual * 100.0) / 100.0
        phred_gq = min(60, -10 * np.log10(1 - gq) if 1 - gq >= 0.0000001 else 60)
        phred_gq = math.ceil(phred_gq * 100.0) / 100.0

        return chrm, st_pos, end_pos, ref, [rec_alt1, rec_alt2], genotype_list[index], phred_qual, phred_gq

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
    def get_proper_alleles(record):
        chrm, st_pos, end_pos, ref, alt_field, genotype, phred_qual, phred_gq = record
        refined_gt = '0/0'
        if genotype == 1:
            refined_gt = '0/1'
        if genotype == 2:
            if len(alt_field) == 1:
                refined_gt = '1/1'
            else:
                refined_gt = '1/2'

        gts = refined_gt.split('/')
        refined_alt = []
        if gts[0] == '1' or gts[1] == '1':
            refined_alt.append(alt_field[0])
        if gts[0] == '2' or gts[1] == '2':
            refined_alt.append(alt_field[1])
        if gts[0] == '0' and gts[1] == '0':
            refined_alt.append('.')

        end_pos = st_pos + len(ref) - 1
        record = chrm, st_pos, end_pos, ref, refined_alt, refined_gt, phred_qual, phred_gq

        return record

    @staticmethod
    def get_filter(record, last_end):
        chrm, st_pos, end_pos, ref, alt_field, genotype, phred_qual, phred_gq = record
        if st_pos <= last_end:
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
