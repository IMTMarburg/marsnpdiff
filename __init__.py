"""
MarSNPDiff is a very simple and fast differential SNP caller.
It examines two sets of bam files and get's you a list of most likely
single nucleotide polymorphisms. It does not cover indels, fusions etc.

The basic idea is to calculate coverage for each nuclotide at each position,
and calculate a log likelihood for each haplotype.
A sequencer error rate of 1% is assumed for the LL calculation.

This is done for both lanes (A and B), then positions where the maximum LL haplotype
differs are scored as follows.
ll_differing = maxLL_A + maxLL_B
ll_same_haplotype_from_A = maxLL_A + ll_B(argMaxLL_A)
ll_same_haplotype_from_B = maxLL_B + ll_A(argMaxLL_B)
score = ll_differing - max(ll_same_haplotype_from_A, ll_same_haplotype_from_B)

The more positive the score, the better the snp.
Result is a pandas.DataFrame with chr, pos, score and coverage information.
"""

import pysam
import pandas
import _marsnpdiff
import multiprocessing
import random


def find_differing_snps_from_vector(coverage_a, coverage_b, ll_threshold = 50):
    positions, candidate_coverage_a, candidate_coverage_b, scores = _marsnpdiff.score_coverage_differences(coverage_a, coverage_b)
    ok = scores >= ll_threshold
    return {
        'positions': positions[ok],
        'coverageA': candidate_coverage_a[:, ok],
        'coverageB': candidate_coverage_b[:, ok],
        'scores': scores[ok]
    }


def count_coverage_multiple(samfiles, chr, start, stop, quality_threshold):
    """Conut and add the coverage of multiple bam files"""
    coverage = _marsnpdiff.count_coverage(samfiles[0], chr, start, stop, quality_threshold)
    for sf in samfiles[1:]:
        cov2 = _marsnpdiff.count_coverage(sf, chr, start, stop, quality_threshold)
        coverage[0] += cov2[0]
        coverage[1] += cov2[1]
        coverage[2] += cov2[2]
        coverage[3] += cov2[3]
    return coverage


def find_differing_snps(samfiles_a, samfiles_b, chr, start, stop, quality_threshold = 15, ll_threshold = 50):
    """Find actual differences between two sets of bam files.
    @samfiles_a (and _b) may be single pysam.AligmentFile/SamFile, or lists of such - in this case the coverage in each set is added together
    @quality_threshold controls which reads/bases are actually considered.
    @ll_threshold: Log likelihood difference must be above this value for the SNP to be reported
    """
    if hasattr(samfiles_a, 'fetch'):
        samfiles_a == [samfiles_a]
    if hasattr(samfiles_b, 'fetch'):
        samfiles_b == [samfiles_b]
    coverage_a = count_coverage_multiple(samfiles_a, chr, start, stop, quality_threshold)
    coverage_b = count_coverage_multiple(samfiles_b, chr, start, stop, quality_threshold)
    res = find_differing_snps_from_vector(coverage_a, coverage_b, start, ll_threshold)
    res['positions'] += start
    res['chr'] = chr
    return res


def iter_chromosome_chunks(chromosome_lengths):
    """Cut all chromosomes into mangable chunks"""
    chunks = []
    chunk_size = int(5e6)
    for chr in chromosome_lengths:
            for start in xrange(0, chromosome_lengths[chr], chunk_size):
                chunks.append((chr, start, start + chunk_size))
    random.shuffle(chunks)
    return chunks


def find_snps(
        bam_filenames_a,
        bam_filenames_b,
        chromosome_lengths,
        quality_threshold = 15,
        ll_threshold = 50, cores_to_use = 4):

    bams_a = [pysam.AlignmentFile(x) for x in bam_filenames_a]
    bams_b = [pysam.AlignmentFile(x) for x in bam_filenames_b]

    p = multiprocessing.Pool(processes=cores_to_use)
    all_snps_found = p.map(find_differing_snps,
            [[bams_a, bams_b, chr, start, stop, quality_threshold, ll_threshold] for (chr, start, stop) in iter_chromosome_chunks(chromosome_lengths)])
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
        res['A_A'].extend(found['coverageA'][0, :])
        res['A_C'].extend(found['coverageA'][1, :])
        res['A_G'].extend(found['coverageA'][2, :])
        res['A_T'].extend(found['coverageA'][3, :])
        res['B_A'].extend(found['coverageB'][0, :])
        res['B_C'].extend(found['coverageB'][1, :])
        res['B_G'].extend(found['coverageB'][2, :])
        res['B_T'].extend(found['coverageB'][3, :])
    p.close()
    p.join()
    return pandas.DataFrame(res)[['chr', 'pos', 'score', 'A_A', 'B_A', 'A_C', 'B_C',
            'A_G', 'B_G', 'A_T', 'B_T']]


if __name__ == '__main__':
    #TODO: implement command line interface
    print "no command line interface has been implemented so far"
    pass
