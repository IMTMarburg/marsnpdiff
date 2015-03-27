#I have had some doubts about the variant calling being performed by VarSCAN - specifially, some locations 
#where the pileups from pysam and the calls were not lining up apperantly (15:83208727 for example)
#so here I'll try to compare two lanes directly...
import numpy as np
import pysam
import numexpr

import numpy as np
cimport numpy as np
from cpython cimport array
cimport cython

import math

DTYPE = np.uint32
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.uint32_t DTYPE_t

@cython.boundscheck(False) #we do manual bounds checking
def count_coverage(samfile, chr, start, stop, quality_threshold = 15):
    """Count ACGT in a part of a sam file. Return 4 numpy arrays of length = stop - start,
    in order A C G T.
    @quality_threshold is the minimum quality score (in phred) a base has to reach to be counted.
    Reads that are any of BAM_FUNMAP, BAM_FSECONDARY, BAM_FQCFAIL, BAM_FDUP are ignored
    """
    
    cdef int _start = start
    cdef int _stop = stop
    cdef int length = _stop - _start
    cdef np.ndarray[DTYPE_t, ndim=1] count_a = np.zeros((length,), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] count_c = np.zeros((length,), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] count_g = np.zeros((length,), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] count_t = np.zeros((length,), dtype=DTYPE)
    cdef char * seq
    cdef array.array quality
    cdef int qpos
    cdef int refpos
    cdef int c = 0
    cdef int _threshold = quality_threshold
    for read in samfile.fetch(chr, start, stop):
        if (read.flag & (0x4 | 0x100 | 0x200 | 0x400)):
            continue
        seq = read.seq
        quality = read.query_qualities
        for qpos, refpos in read.get_aligned_pairs(True):
            if qpos is not None and refpos is not None and _start <= refpos < _stop:
                if quality[qpos] > quality_threshold:
                    if seq[qpos] == 'A':
                        count_a[refpos - _start] += 1
                    if seq[qpos] == 'C':
                        count_c[refpos - _start] += 1
                    if seq[qpos] == 'G':
                        count_g[refpos - _start] += 1
                    if seq[qpos] == 'T':
                        count_t[refpos - _start] += 1
    return count_a, count_c, count_g, count_t

cdef read_error_prob = 0.001  #retrieved from http://www.molecularecologist.com/next-gen-table-3c-2014/
cdef float ll_99 = math.log(1 - read_error_prob)
cdef float ll_003 = math.log(read_error_prob/3)
cdef float ll_005 = math.log(read_error_prob/2)
cdef float ll_495 = math.log((1-read_error_prob)/2)
cdef float ll_25 = math.log(0.25)

llPosToHaplotype = [ 'AA', 'AC','AG','AT', 'CC', 'CG','CT','GG', 'GT', 'TT', 'NN']

cdef _logLikelihood2(np.ndarray count_a, np.ndarray count_c, np.ndarray count_g, np.ndarray count_t):
    res = np.zeros((11, count_a.shape[0]), dtype=np.float)
    d = {'count_a': count_a, 'count_c': count_c, 'count_g': count_g, 'count_t': count_t,
            'll_99': ll_99, 'll_003': ll_003, 'll_005': ll_005, 'll_495': ll_495, 'll_25': ll_25}
    count_a__ll_003 = count_a * ll_003
    count_c__ll_003 = count_c * ll_003
    count_g__ll_003 = count_g * ll_003
    count_t__ll_003 = count_t * ll_003
    count_a__ll_005 = count_a * ll_005
    count_c__ll_005 = count_c * ll_005
    count_g__ll_005 = count_g * ll_005
    count_t__ll_005 = count_t * ll_005
    count_a__ll_495 = count_a * ll_495
    count_c__ll_495 = count_c * ll_495
    count_g__ll_495 = count_g * ll_495
    count_t__ll_495 = count_t * ll_495

    d = {'count_a': count_a, 'count_c': count_c, 'count_g': count_g, 'count_t': count_t,
         'count_a__ll_003': count_a__ll_003, 
         'count_c__ll_003': count_c__ll_003, 
         'count_g__ll_003': count_g__ll_003, 'count_t__ll_003': count_t__ll_003, 'count_a__ll_005': count_a__ll_005, 'count_c__ll_005': count_c__ll_005, 'count_g__ll_005': count_g__ll_005, 'count_t__ll_005': count_t__ll_005, 'count_a__ll_495': count_a__ll_495, 'count_c__ll_495': count_c__ll_495, 'count_g__ll_495': count_g__ll_495, 'count_t__ll_495': count_t__ll_495,
        'll_99': ll_99, 'll_003': ll_003, 'll_005': ll_005, 'll_495': ll_495, 'll_25': ll_25}
    res[0,:] = numexpr.evaluate("(count_a * ll_99 + count_c__ll_003 + count_g__ll_003 + count_t__ll_003)",d)#, 'AA'), 0
    res[1,:] = numexpr.evaluate("(count_a__ll_495 + count_c__ll_495 + count_g__ll_005 + count_t__ll_005)",d)#, 'AC'),1
    res[2,:] = numexpr.evaluate("(count_a__ll_495 + count_c__ll_005 + count_g__ll_495 + count_t__ll_005)",d)#, 'AG'),2
    res[3,:] = numexpr.evaluate("(count_a__ll_495 + count_c__ll_005 + count_g__ll_005 + count_t__ll_495)",d)#, 'AT'),3
    res[4,:] = numexpr.evaluate("(count_a__ll_003 + count_c * ll_99 + count_g__ll_003 + count_t__ll_003)",d)#, 'CC'), 4
    res[5,:] = numexpr.evaluate("(count_a__ll_005 + count_c__ll_495 + count_g__ll_495 + count_t__ll_005)",d)#, 'CG'),5
    res[6,:] = numexpr.evaluate("(count_a__ll_005 + count_c__ll_495 + count_g__ll_005 + count_t__ll_495)",d)#, 'CT'),6
    res[7,:] = numexpr.evaluate("(count_a__ll_005 + count_c__ll_005 + count_g * ll_99 + count_t__ll_005)",d)#, 'GG'), 7
    res[8,:] = numexpr.evaluate("(count_a__ll_005 + count_c__ll_005 + count_g__ll_495 + count_t__ll_495)",d)#, 'GT'), 8
    res[9,:] = numexpr.evaluate("(count_a__ll_003 + count_c__ll_003 + count_g__ll_003 + count_t * ll_99)",d)#, 'TT'), 9
    res[10,:] = numexpr.evaluate("(count_a * ll_25 + count_c * ll_25 + count_g * ll_25 + count_t * ll_25)",d)#, 'NN'), 10
    return res

def logLikelihood(count_a_or_dict, count_c=None, count_g=None, count_t=None):
    """Calculate the log likelihood for AA, AC, AT, CC... etc.
    Assumption is a total sequencing (SNP) error rate of 1%
    See llPosToHaplotype for order
    """
    if isinstance(count_a_or_dict, dict):
        count_c = count_a_or_dict['C']
        count_g = count_a_or_dict['G']
        count_t = count_a_or_dict['T']
        count_a_or_dict = count_a_or_dict['A']
    return _logLikelihood2(count_a_or_dict, count_c, count_g, count_t)

def score_coverage_differences(coverage_a, coverage_b):
    llA = logLikelihood(*coverage_a)
    llB = logLikelihood(*coverage_b)
    coverage_a = np.array(coverage_a)
    coverage_b = np.array(coverage_b)
    llMaxA = llA.max(axis=0)
    llMaxB = llB.max(axis=0)
    llArgMaxA = llA.argmax(axis=0)
    llArgMaxB = llB.argmax(axis=0)
    llArgMaxA[(llMaxA == 0) | (llMaxB == 0)] = 99
    llArgMaxB[(llMaxA == 0) | (llMaxB == 0)] = 99
    #candidates are all where the max LL derived haplotype is not the same
    candidates = np.where(llArgMaxA != llArgMaxB)[0]
    haplotypeA = llArgMaxA[candidates]
    haplotypeB = llArgMaxB[candidates]
    best_llA = llA[:,candidates][[haplotypeA, xrange(0, len(candidates))]]
    best_llB = llB[:,candidates][[haplotypeB, xrange(0, len(candidates))]]

    ll_differing = best_llA + best_llB
    ll_same_haplotypeA = best_llA + \
            llB[:,candidates][[haplotypeA, xrange(0, len(candidates))]]
    ll_same_haplotypeB = llA[:,candidates][[haplotypeB, xrange(0, len(candidates))]] + \
            best_llB
    ll_same_max = np.array([ll_same_haplotypeA, ll_same_haplotypeB]).max(axis=0)
        

    score = ll_differing - ll_same_max
    return (candidates, 
            #llA[:,candidates], llB[:,candidates], llArgMaxA[candidates], llArgMaxB[candidates], \
            coverage_a[:,candidates], 
            coverage_b[:,candidates],
            score,
            )


def find_differing_snps(samfile_a, samfile_b, chr, start, stop, quality_threshold = 15, ll_threshold = 50):
    """Find actual differences between two bam files.
    @quality_threshold controls which reads/bases are actually considered.
    @ll_threshold: Log likelihood difference must be above this value for the SNP to be reported
    """
    coverage_a = count_coverage(samfile_a, chr, start, stop, quality_threshold)
    coverage_b = count_coverage(samfile_b, chr, start, stop, quality_threshold)
    positions, coverage_a , coverage_b, scores = score_coverage_differences(coverage_a, coverage_b)
    ok = scores >= ll_threshold
    return {
            'chr': chr,
            'positions': positions[ok] + start, 
            'coverageA': coverage_a[:,ok],
            'coverageB': coverage_b[:,ok],
            'scores': scores[ok]
            }
    
