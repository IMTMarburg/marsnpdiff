"""
MarSNPDiff is a very simple and fast differential SNP caller.
It examines two bam files and get's you a list of most likely 
single nucleotide polymorphisms. It does not cover indels, fusions etc.

The basic idea is to calculate coverage for each nuclotide at each position,
and calculate a log likelihood for each haplotype.
A sequencer error rate of 1% is assumed for the LL calculation.

This is done for both lanes (A and B) , then positions where the maximum LL haplotype
differs are scored as follows.
ll_differing = maxLL_A + maxLL_B
ll_same_haplotype_from_A = maxLL_A + ll_B(argMaxLL_A)
ll_same_haplotype_from_B = maxLL_B + ll_A(argMaxLL_B)
score = ll_differing - max(ll_same_haplotype_from_A, ll_same_haplotype_from_B)

The more positive the score, the better the snp.
Result is a DataFrame with chr, pos, score and coverage information.
"""

import pysam
import pandas
import pyximport
pyximport.install()
from _marsnpdiff import find_differing_snps
import multiprocessing
import random

def iter_chromosome_chunks(chromosome_lengths):
    chunks = []
    chunk_size = int(5e6)
    for chr in chromosome_lengths:
            for start in xrange(0, chromosome_lengths[chr], chunk_size):
                chunks.append( (chr, start, start + chunk_size))
    random.shuffle(chunks)
    return chunks

def call_find_differing_snps(args):
    args[0] = pysam.AlignmentFile(args[0])
    args[1] = pysam.AlignmentFile(args[1])
    return find_differing_snps(*args)

def find_snps(
        bam_filename_a,
        bam_filename_b,
        chromosome_lengths,
        quality_threshold = 15,
        ll_threshold = 50):

    p = multiprocessing.Pool(processes=5)
    all_snps_found = p.map(call_find_differing_snps,
            [[bam_filename_a, bam_filename_b, chr, start, stop, quality_threshold, ll_threshold] for (chr, start, stop) in iter_chromosome_chunks(chromosome_lengths)])
    res = {'chr': [], 'pos': [], 'score': [],
            'A_A': [], 
            'A_C': [], 
            'A_G': [], 
            'A_T': [], 
            'B_A': [], 
            'B_C': [], 
            'B_G': [], 
            'B_T': [], }
    for found in all_snps_found:
        res['chr'].extend([found['chr']] * len(found['positions']))
        res['pos'].extend(found['positions'])
        res['score'].extend(found['scores'])
        res['A_A'].extend(found['coverageA'][0,:])
        res['A_C'].extend(found['coverageA'][1,:])
        res['A_G'].extend(found['coverageA'][2,:])
        res['A_T'].extend(found['coverageA'][3,:])
        res['B_A'].extend(found['coverageB'][0,:])
        res['B_C'].extend(found['coverageB'][1,:])
        res['B_G'].extend(found['coverageB'][2,:])
        res['B_T'].extend(found['coverageB'][3,:])
    p.close()
    p.join()
    return pandas.DataFrame(res)[[ 'chr', 'pos', 'score', 'A_A', 'B_A', 'A_C', 'B_C',
            'A_G', 'B_G', 'A_T', 'B_T',]]


if __name__ == '__main__': 
    a = ( "../../results/AlignedLane/OC66_gr40__aligned_with_STAR_against_EnsemblGenome_Homo_sapiens_74_37/aligned_unique_OC66_gr40__aligned_with_STAR_against_EnsemblGenome_Homo_sapiens_74_37.bam")
    b = ("../../results/AlignedLane/OC65_gr40__aligned_with_STAR_against_EnsemblGenome_Homo_sapiens_74_37/aligned_unique_OC65_gr40__aligned_with_STAR_against_EnsemblGenome_Homo_sapiens_74_37.bam")
    chr_lengths = pandas.read_csv('/local/ensembl_track3/Homo_sapiens_74/lookup/chromosome_lengths.tsv', sep="\t", header=None)
    chr_lengths = dict(zip(chr_lengths[0], chr_lengths[1]))
    print find_snps(a, b, chr_lengths).sort(['score'], ascending=False).reset_index(drop=True)
    #TODO: implement command line interface
        







