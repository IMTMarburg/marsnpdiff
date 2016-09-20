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
import scipy.linalg.blas as blas

DTYPE = np.uint32
ctypedef np.uint32_t DTYPE_t

DTYPE_flt = np.float32
ctypedef np.float32_t DTYPE_flt_t

DTYPE_byte = np.uint8
ctypedef np.uint8_t DTYPE_byte_t

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
    cdef np.ndarray[DTYPE_flt_t, ndim=1] count_a = np.zeros((length,), dtype=DTYPE_flt)
    cdef np.ndarray[DTYPE_flt_t, ndim=1] count_c = np.zeros((length,), dtype=DTYPE_flt)
    cdef np.ndarray[DTYPE_flt_t, ndim=1] count_g = np.zeros((length,), dtype=DTYPE_flt)
    cdef np.ndarray[DTYPE_flt_t, ndim=1] count_t = np.zeros((length,), dtype=DTYPE_flt)
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
#                      0     1    2    3     4     5    6    7    8      9    10

def cpy_sscal(factor, vector):
    """Convert blas.sscall into a non-input modifying variant"""
    result = vector.copy()
    blas.sscal(factor, result)
    return result

cdef _logLikelihood2(np.ndarray[DTYPE_flt_t, ndim=1] count_a, np.ndarray[DTYPE_flt_t, ndim=1] count_c, np.ndarray[DTYPE_flt_t, ndim=1] count_g, np.ndarray[DTYPE_flt_t, ndim=1] count_t):
    res = []
    cdef np.ndarray[DTYPE_flt_t, ndim=1] count_a__ll_003 = cpy_sscal(ll_003, count_a) #count_a * ll_003
    cdef np.ndarray[DTYPE_flt_t, ndim=1] count_c__ll_003 = cpy_sscal(ll_003, count_c)
    cdef np.ndarray[DTYPE_flt_t, ndim=1] count_g__ll_003 = cpy_sscal(ll_003, count_g)
    cdef np.ndarray[DTYPE_flt_t, ndim=1] count_t__ll_003 = cpy_sscal(ll_003, count_t)
    cdef np.ndarray[DTYPE_flt_t, ndim=1] count_a__ll_005 = cpy_sscal(ll_005, count_a)
    cdef np.ndarray[DTYPE_flt_t, ndim=1] count_c__ll_005 = cpy_sscal(ll_005, count_c)
    cdef np.ndarray[DTYPE_flt_t, ndim=1] count_g__ll_005 = cpy_sscal(ll_005, count_g)
    cdef np.ndarray[DTYPE_flt_t, ndim=1] count_t__ll_005 = cpy_sscal(ll_005, count_t)
    cdef np.ndarray[DTYPE_flt_t, ndim=1] count_a__ll_495 = cpy_sscal(ll_495, count_a)
    cdef np.ndarray[DTYPE_flt_t, ndim=1] count_c__ll_495 = cpy_sscal(ll_495, count_c)
    cdef np.ndarray[DTYPE_flt_t, ndim=1] count_g__ll_495 = cpy_sscal(ll_495, count_g)
    cdef np.ndarray[DTYPE_flt_t, ndim=1] count_t__ll_495 = cpy_sscal(ll_495, count_t)

    cdef np.ndarray[DTYPE_flt_t, ndim=1] temp1

    #res[0,:] = numexpr.evaluate("(count_a * ll_99 + count_c__ll_003 + count_g__ll_003 + count_t__ll_003)",d)#, 'AA'), 0
    temp1 = cpy_sscal(ll_99, count_a)
    blas.saxpy(count_c__ll_003, temp1)
    blas.saxpy(count_g__ll_003, temp1)
    blas.saxpy(count_t__ll_003, temp1)
    res.append(temp1.copy())

    #res[1,:] = numexpr.evaluate("(count_a__ll_495 + count_c__ll_495 + count_g__ll_005 + count_t__ll_005, temp1)",d)#, 'AC'),1
    temp1 = count_a__ll_495.copy()
    blas.saxpy(count_c__ll_495, temp1)
    blas.saxpy(count_g__ll_005, temp1)
    blas.saxpy(count_t__ll_005, temp1)
    res.append(temp1.copy())

    #res[2,:] = numexpr.evaluate("(count_a__ll_495 + count_c__ll_005 + count_g__ll_495 + count_t__ll_005, temp1)",d)#, 'AG'),2
    temp1 = count_a__ll_495.copy()
    blas.saxpy(count_c__ll_005, temp1)
    blas.saxpy(count_g__ll_495, temp1)
    blas.saxpy(count_t__ll_005, temp1)
    res.append(temp1.copy())

    #res[3,:] = numexpr.evaluate("(count_a__ll_495 + count_c__ll_005 + count_g__ll_005 + count_t__ll_495, temp1)",d)#, 'AT'),3
    temp1 = count_a__ll_495.copy()
    blas.saxpy(count_c__ll_005, temp1)
    blas.saxpy(count_g__ll_005, temp1)
    blas.saxpy(count_t__ll_495, temp1)
    res.append(temp1.copy())

    #res[4,:] = numexpr.evaluate("(count_a__ll_003 + count_c * ll_99 + count_g__ll_003 + count_t__ll_003, temp1)",d)#, 'CC'), 4
    temp1 = cpy_sscal(ll_99, count_c)
    blas.saxpy(count_a__ll_003, temp1)
    blas.saxpy(count_g__ll_003, temp1)
    blas.saxpy(count_t__ll_003, temp1)
    res.append(temp1.copy())

    #res[5,:] = numexpr.evaluate("(count_a__ll_005 + count_c__ll_495 + count_g__ll_495 + count_t__ll_005, temp1)",d)#, 'CG'),5
    temp1 = count_a__ll_005.copy()
    blas.saxpy(count_c__ll_495, temp1)
    blas.saxpy(count_g__ll_495, temp1)
    blas.saxpy(count_t__ll_005, temp1)
    res.append(temp1.copy())

    #res[6,:] = numexpr.evaluate("(count_a__ll_005 + count_c__ll_495 + count_g__ll_005 + count_t__ll_495, temp1)",d)#, 'CT'),6
    temp1 = count_a__ll_005.copy()
    blas.saxpy(count_c__ll_495, temp1)
    blas.saxpy(count_g__ll_005, temp1)
    blas.saxpy(count_t__ll_495, temp1)
    res.append(temp1.copy())

    #res[7,:] = numexpr.evaluate("(count_a__ll_005 + count_c__ll_005 + count_g * ll_99 + count_t__ll_005, temp1)",d)#, 'GG'), 7
    temp1 = cpy_sscal(ll_99, count_g)
    blas.saxpy(count_a__ll_003, temp1)
    blas.saxpy(count_c__ll_003, temp1)
    blas.saxpy(count_t__ll_003, temp1)
    res.append(temp1.copy())

    #res[8,:] = numexpr.evaluate("(count_a__ll_005 + count_c__ll_005 + count_g__ll_495 + count_t__ll_495, temp1)",d)#, 'GT'), 8
    temp1 = count_a__ll_005.copy()
    blas.saxpy(count_c__ll_005, temp1)
    blas.saxpy(count_g__ll_495, temp1)
    blas.saxpy(count_t__ll_495, temp1)
    res.append(temp1.copy())

    #res[9,:] = numexpr.evaluate("(count_a__ll_003 + count_c__ll_003 + count_g__ll_003 + count_t * ll_99, temp1)",d)#, 'TT'), 9
    temp1 = cpy_sscal(ll_99, count_t)
    blas.saxpy(count_c__ll_003, temp1)
    blas.saxpy(count_g__ll_003, temp1)
    blas.saxpy(count_a__ll_003, temp1)
    res.append(temp1.copy())

    #res[10,:] = numexpr.evaluate("(count_a * ll_25 + count_c * ll_25 + count_g * ll_25 + count_t * ll_25)",d)#, 'NN'), 10
    temp1 = cpy_sscal(ll_25, count_a)
    blas.saxpy(cpy_sscal(ll_25, count_c), temp1)
    blas.saxpy(cpy_sscal(ll_25, count_g), temp1)
    blas.saxpy(cpy_sscal(ll_25, count_t), temp1)
    res.append(temp1.copy())

    del count_a__ll_003
    del count_c__ll_003
    del count_g__ll_003
    del count_t__ll_003
    del count_a__ll_005
    del count_c__ll_005
    del count_g__ll_005
    del count_t__ll_005
    del count_a__ll_495
    del count_c__ll_495
    del count_g__ll_495
    del count_t__ll_495
    del temp1
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


cdef cmpMax(DTYPE_byte_t* argMax, DTYPE_flt_t* valueMax, DTYPE_flt_t* A, DTYPE_flt_t* B, unsigned int length, DTYPE_byte_t arg_pos):
    for i in range(length):
        if B[i] > A[i]:
            argMax[i] = arg_pos
            valueMax[i] = B[i]


def llMax(ll):
    """calculate max and argmax from the result of logLikelihood"""
    cdef np.ndarray[DTYPE_flt_t, ndim=1] AA = ll[0]
    cdef np.ndarray[DTYPE_flt_t, ndim=1] AC = ll[1]
    cdef np.ndarray[DTYPE_flt_t, ndim=1] AG = ll[2]
    cdef np.ndarray[DTYPE_flt_t, ndim=1] AT = ll[3]
    cdef np.ndarray[DTYPE_flt_t, ndim=1] CC = ll[4]
    cdef np.ndarray[DTYPE_flt_t, ndim=1] CG = ll[5]
    cdef np.ndarray[DTYPE_flt_t, ndim=1] CT = ll[6]
    cdef np.ndarray[DTYPE_flt_t, ndim=1] GG = ll[7]
    cdef np.ndarray[DTYPE_flt_t, ndim=1] GT = ll[8]
    cdef np.ndarray[DTYPE_flt_t, ndim=1] TT = ll[9]
    cdef np.ndarray[DTYPE_flt_t, ndim=1] NN = ll[10]

    cdef unsigned int length = AA.shape[0]
    cdef np.ndarray[DTYPE_byte_t, ndim=1] argMax = np.zeros((length,), DTYPE_byte)
    cdef np.ndarray[DTYPE_flt_t, ndim=1] valueMax = np.full((length,), np.finfo(DTYPE_flt).min, DTYPE_flt)
    cmpMax(&argMax[0], &valueMax[0], &valueMax[0], &AA[0], length, 0) 
    cmpMax(&argMax[0], &valueMax[0], &valueMax[0], &AC[0], length, 1) 
    cmpMax(&argMax[0], &valueMax[0], &valueMax[0], &AG[0], length, 2) 
    cmpMax(&argMax[0], &valueMax[0], &valueMax[0], &AT[0], length, 3) 
    cmpMax(&argMax[0], &valueMax[0], &valueMax[0], &CC[0], length, 4) 
    cmpMax(&argMax[0], &valueMax[0], &valueMax[0], &CG[0], length, 5) 
    cmpMax(&argMax[0], &valueMax[0], &valueMax[0], &CT[0], length, 6) 
    cmpMax(&argMax[0], &valueMax[0], &valueMax[0], &GG[0], length, 7) 
    cmpMax(&argMax[0], &valueMax[0], &valueMax[0], &GT[0], length, 8) 
    cmpMax(&argMax[0], &valueMax[0], &valueMax[0], &TT[0], length, 9) 
    cmpMax(&argMax[0], &valueMax[0], &valueMax[0], &NN[0], length, 10) 
    return valueMax, argMax


def score_coverage_differences(coverage_a, coverage_b):
    llA = logLikelihood(*coverage_a)
    llB = logLikelihood(*coverage_b)
    llMaxA, llArgMaxA = llMax(llA)
    llMaxB, llArgMaxB = llMax(llB)
    
    llArgMaxA[(llMaxA == 0) | (llMaxB == 0)] = 99
    llArgMaxB[(llMaxA == 0) | (llMaxB == 0)] = 99
    # candidates are all where the max LL derived haplotype is not the same
    # filtering by score happens later on
    candidates = np.where(llArgMaxA != llArgMaxB)[0]
    haplotypeA = llArgMaxA[candidates]
    haplotypeB = llArgMaxB[candidates]
    best_llA = []
    for ii, candidate_pos in enumerate(candidates):
        best_llA.append(llA[haplotypeA[ii]][candidate_pos])
    best_llB = []
    for ii, candidate_pos in enumerate(candidates):
        best_llB.append(llB[haplotypeB[ii]][candidate_pos])
    second_best_llA = []
    for ii, candidate_pos in enumerate(candidates):
        second_best_llA .append(llA[haplotypeB[ii]][candidate_pos])

    second_best_llB = []
    for ii, candidate_pos in enumerate(candidates):
        second_best_llB .append(llB[haplotypeA[ii]][candidate_pos])

    best_llA = np.array(best_llA)
    best_llB = np.array(best_llB)
    second_best_llA = np.array(second_best_llA)
    second_best_llB = np.array(second_best_llB)

    ll_differing = best_llA + best_llB
    ll_same_haplotypeA = best_llA + second_best_llB
    ll_same_haplotypeB = second_best_llA +  best_llB
    ll_same_max = np.array([ll_same_haplotypeA, ll_same_haplotypeB]).max(axis=0)
    score = ll_differing - ll_same_max


    result_cov_a = [
            coverage_a[0][candidates],
            coverage_a[1][candidates],
            coverage_a[2][candidates],
            coverage_a[3][candidates],
    ]
    result_cov_b = [
            coverage_b[0][candidates],
            coverage_b[1][candidates],
            coverage_b[2][candidates],
            coverage_b[3][candidates],
    ]
    return (candidates, 
            #llA[:,candidates], llB[:,candidates], llArgMaxA[candidates], llArgMaxB[candidates], \
            result_cov_a,
            result_cov_b,
            score,
            )
