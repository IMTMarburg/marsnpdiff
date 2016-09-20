import unittest
import sys
sys.path.append('../pysam')
import _marsnpdiff
import math
import numpy as np
import numexpr

read_error_prob = 0.001  # retrieved from http://www.molecularecologist.com/next-gen-table-3c-2014/
ll_99 = math.log(1 - read_error_prob)
ll_003 = math.log(read_error_prob / 3)
ll_005 = math.log(read_error_prob / 2)
ll_495 = math.log((1 - read_error_prob) / 2)
ll_25 = math.log(0.25)


def _logLikelihood2(count_a, count_c, count_g, count_t):
    res = np.zeros((11, count_a.shape[0]), dtype=np.float)
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
    res[0, :] = numexpr.evaluate("(count_a * ll_99 + count_c__ll_003 + count_g__ll_003 + count_t__ll_003)", d)  # 'AA'), 0
    res[1, :] = numexpr.evaluate("(count_a__ll_495 + count_c__ll_495 + count_g__ll_005 + count_t__ll_005)", d)  # 'AC'),1
    res[2, :] = numexpr.evaluate("(count_a__ll_495 + count_c__ll_005 + count_g__ll_495 + count_t__ll_005)", d)  # 'AG'),2
    res[3, :] = numexpr.evaluate("(count_a__ll_495 + count_c__ll_005 + count_g__ll_005 + count_t__ll_495)", d)  # 'AT'),3
    res[4, :] = numexpr.evaluate("(count_a__ll_003 + count_c * ll_99 + count_g__ll_003 + count_t__ll_003)", d)  # 'CC'), 4
    res[5, :] = numexpr.evaluate("(count_a__ll_005 + count_c__ll_495 + count_g__ll_495 + count_t__ll_005)", d)  # 'CG'),5
    res[6, :] = numexpr.evaluate("(count_a__ll_005 + count_c__ll_495 + count_g__ll_005 + count_t__ll_495)", d)  # 'CT'),6
    res[7, :] = numexpr.evaluate("(count_a__ll_005 + count_c__ll_005 + count_g * ll_99 + count_t__ll_005)", d)  # 'GG'), 7
    res[8, :] = numexpr.evaluate("(count_a__ll_005 + count_c__ll_005 + count_g__ll_495 + count_t__ll_495)", d)  # 'GT'), 8
    res[9, :] = numexpr.evaluate("(count_a__ll_003 + count_c__ll_003 + count_g__ll_003 + count_t * ll_99)", d)  # 'TT'), 9
    res[10, :] = numexpr.evaluate("(count_a * ll_25 + count_c * ll_25 + count_g * ll_25 + count_t * ll_25)", d)  # 'NN'), 10
    return res


class LLTests(unittest.TestCase):
    def test_ll(self):
        count_a = [100, 0, 0, 100, 25]
        count_c = [0, 200, 0, 0, 25]
        count_g = [0, 0, 100, 0, 25]
        count_t = [0, 0, 0, 100, 25]
        count_a = np.array(count_a, dtype=np.float32)
        count_c = np.array(count_c, dtype=np.float32)
        count_g = np.array(count_g, dtype=np.float32)
        count_t = np.array(count_t, dtype=np.float32)
        ll = _marsnpdiff.logLikelihood(count_a, count_c, count_g, count_t)
        should = _logLikelihood2(count_a, count_c, count_g, count_t)
        should = should.astype(np.float32)
        self.assertEqual(len(ll), 11)
        self.assertEqual(len(ll[0]), 5)
        for p in xrange(5):
            # print p
            for i in xrange(0, 11):
                # print i,
                # if abs(ll[i][p]- should[i][p]) > 0.0001:
                    # print '!!!',
                # print "%.15f" % ll[i][p],"%.15f" % should[i][p]
                # #print "%.15f" % round(ll[i][p]-should[i][p], 3)
                self.assertAlmostEquals(ll[i][p], should[i][p], 3)

        self.assertAlmostEqual(ll[0][0], 100 * ll_99)
        self.assertAlmostEqual(ll[4][1], 200 * ll_99)
        self.assertAlmostEqual(ll[7][2], 100 * ll_99)
        self.assertAlmostEqual(ll[3][3], 100 * ll_495 * 2, 4)

    def test_llMax(self):
        input = [
                [-1, 0, 0, -1, -1, -5],  # j0
                [-2, 0, 0, -0.01, -2.1, -6],  # 1
                [-4, 0, 0, -1, -0.5, -7],  # 2
                [-5, 0, 0, -1, -1, -1],  # 3
                [-1, 0, 0, -1, -1, -5],  # 4
                [-2, 0, 0, -1, -1, -100],  # 5
                [-2, 0, 0, -1, -1, -12],  # 6
                [-2, 0, 0, -1, -1, -23],  # 7
                [-2, 0, 0, -1, -1, -1.3],  # 8
                [-2, 0, 10, -1, -1, -1.11],  # 9
                [-2, 0, 0, -1, -1, -1.0001],  # 10
        ]
        input = [np.array(x, dtype=np.float32) for x in input]
        valueMax, argMax = _marsnpdiff.llMax(input)
        self.assertTrue((np.array([0, 0, 9, 1, 2, 3]) == argMax).all())
        self.assertFalse(
                (np.abs(np.array([-1, 0, 10, -0.01, -0.5, -1]) - valueMax) > 0.0001).any())


class ScoreTests(unittest.TestCase):
    def test_simple(self):
        coverage_a = (
                np.array([100, 0, 0, 25], dtype=np.float32),
                np.array([0, 100, 0, 25], dtype=np.float32),
                np.array([0, 0, 100, 25], dtype=np.float32),
                np.array([0, 0, 100, 25], dtype=np.float32),

        )
        coverage_b = (
                np.array([0, 0, 100, 50], dtype=np.float32),
                np.array([100, 0, 0, 50], dtype=np.float32),
                np.array([0, 0, 0, 0], dtype=np.float32),
                np.array([0, 0, 100, 0], dtype=np.float32),
        )
        candidates, ccov_a, ccov_b, scores = _marsnpdiff.score_coverage_differences(coverage_a, coverage_b)
        self.assertTrue((candidates == [0, 2, 3]).all())
        self.assertTrue((ccov_a[0] == [100, 0, 25]).all())
        self.assertTrue((ccov_a[1] == [0, 0, 25]).all())
        self.assertTrue((ccov_a[2] == [0, 100, 25]).all())
        self.assertTrue((ccov_a[3] == [0, 100, 25]).all())

        self.assertTrue((ccov_b[0] == [0, 100, 50]).all())
        self.assertTrue((ccov_b[1] == [100, 0, 50]).all())
        self.assertTrue((ccov_b[2] == [0, 0, 0]).all())
        self.assertTrue((ccov_b[3] == [0, 100, 0]).all())


if __name__ == '__main__':
    unittest.main()
