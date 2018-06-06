import pysam
import sys

file_path = sys.argv[1]
vcf_records = pysam.VariantFile(file_path)

total_indels = 0
total_dalts_indels = 0
total_salts_indels = 0

total_snps = 0
total_dalts_snps = 0
total_salts_snps = 0
for record in vcf_records:
    # print(record)
    gts = [s['GT'] for s in record.samples.values()]
    gt = gts[0]
    lens = [len(x) for x in record.alleles]

    if max(lens) > 1 and len(gt) == 2 and gt[0] != 0 and gt[1] != 0 and gt[0] != gt[1]:
        print(record)
        # print(record.qual)
        total_indels += 1
        total_dalts_indels += 1
    elif max(lens) > 1:
        total_indels += 1
        total_salts_indels += 1

    if max(lens) == 1 and len(gt) == 2 and gt[0] != 0 and gt[1] != 0 and gt[0] != gt[1]:
        # print(record)
        total_snps += 1
        total_dalts_snps += 1
    elif max(lens) == 1:
        # print(record)
        total_snps += 1
        total_salts_snps += 1

print("TOTAL INDELS: ", total_indels)
print("TOTAL DBALTS: ", total_dalts_indels, total_dalts_indels/total_indels * 100)
print("TOTAL SNALTS: ", total_salts_indels, total_salts_indels/total_indels * 100)
print("---------------------------------------------------")
print("TOTAL SNPS: ", total_snps)
print("TOTAL DALTS: ", total_dalts_snps, total_dalts_snps/total_snps * 100)
print("TOTAL SALTS: ", total_salts_snps, total_salts_snps/total_snps * 100)
