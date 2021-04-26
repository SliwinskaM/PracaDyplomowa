import unittest
import apriori as apr
import association_rules_division as ard
import association_rules_pure_python as aprpp
from additional_functions import create_converted_r_matrix
import fuzzy_curves as fc
import recommend as re
import visualizations as vs
import numpy as np


class TestRecommendation(unittest.TestCase):
    def test_single_user(self):
        test_conv_r_matrix = np.empty((1, 3, 3))
        test_conv_r_matrix[:] = np.nan
        test_conv_r_matrix[0, 0] = np.array((0, 0, 1))
        test_conv_r_matrix[0, 1] = np.array((0, 0, 1))
        test_conv_r_matrix[0, 2] = np.array((0, 0, 1))

        test_rules = [[[[0, 2], [1, 2]], [[2, 2]]]]

        test_matrix = np.empty((1, 3, 3))
        test_matrix[:] = np.nan
        test_matrix[0, 0] = np.array((0, 0, 1))
        test_matrix[0, 1] = np.array((0, 0, 1))

        recom = re.Recommend(test_conv_r_matrix)
        r = recom.recommend_to_user(test_rules, test_matrix, 0)
        self.assertEqual(r, [np.array([2])])

    def test_whole(self):
        test_conv_r_matrix = np.empty((2, 3, 3))
        test_conv_r_matrix[:] = np.nan
        test_conv_r_matrix[0, 0] = np.array((0, 0, 1))
        test_conv_r_matrix[0, 1] = np.array((0, 0, 1))
        test_conv_r_matrix[0, 2] = np.array((0, 0, 1))
        test_conv_r_matrix[1, 0] = np.array((0, 0, 1))
        test_conv_r_matrix[1, 1] = np.array((0, 0, 1))

        curves = fc.Curves1(1, 5, 0.2, 0.45, 0.55, 0.8)

        recom = re.Recommend(test_conv_r_matrix)
        tr = recom.main_recommend(100, curves.Names, cross_num=3, test_size=0.3, shuffle_test=False, min_support=0.0000000001, min_confidence=0.000004)
        self.assertEqual(tr, ([[np.array([0], dtype=object), np.array([2], dtype=object)], # 1 -> 0 and 2
                                [np.array([1], dtype=object), np.array([2], dtype=object)], # 0 -> 1 and 2
                                  [np.array([2], dtype=object), np.array([2], dtype=object)]], # 0 and 1 -> 2 and 2
                                 0.3333333333333333))



    def test_whole_identical(self):
        test_conv_r_matrix = np.empty((2, 3, 3))
        test_conv_r_matrix[:] = np.nan
        test_conv_r_matrix[0, 0] = np.array((0, 0, 1))
        test_conv_r_matrix[0, 1] = np.array((0, 0, 1))
        test_conv_r_matrix[0, 2] = np.array((0, 0, 1))
        test_conv_r_matrix[1, 0] = np.array((0, 0, 1))
        test_conv_r_matrix[1, 1] = np.array((0, 0, 1))
        test_conv_r_matrix[1, 2] = np.array((0, 0, 1))

        curves = fc.Curves1(1, 5, 0.2, 0.45, 0.55, 0.8)

        recom = re.Recommend(test_conv_r_matrix)
        tr = recom.main_recommend(100, curves.Names, cross_num=3, test_size=0.3, shuffle_test=False, min_support=0.0000000001, min_confidence=0.000004)
        # every time both products in base recommend the third in test
        self.assertEqual(tr, ([[np.array([0], dtype=object), np.array([0], dtype=object)],
                                [np.array([1], dtype=object), np.array([1], dtype=object)],
                                  [np.array([2], dtype=object), np.array([2], dtype=object)]],
                                 1.0))

if __name__ == '__main__':
    unittest.main()