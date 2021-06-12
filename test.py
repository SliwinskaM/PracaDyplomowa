import unittest

import import_data
import apriori as apr
import association_rules_division as ard
import association_rules_pure_python as aprpp
from additional_functions import create_converted_r_matrix
import fuzzy_curves as fc
import recommend as re
import visualizations as vs
import numpy as np

#proba

class TestAll(unittest.TestCase):
    def test_apriori(self):
        data = import_data.ImportData('test')
        data.import_data()
        curves = fc.Curves1(data.min_score, data.max_score, 0.2, 0.45, 0.55, 0.8)
        conv_r_matrix = create_converted_r_matrix(data.r_matrix, curves)
        conv_r_matrix_wanted = np.empty((5, 4, 3))
        conv_r_matrix_wanted[:] = np.nan
        conv_r_matrix_wanted[0, 0] = np.array((0, 0, 1))
        conv_r_matrix_wanted[0, 1] = np.array((0, 0, 1))
        conv_r_matrix_wanted[0, 2] = np.array((0, 0, 1))
        conv_r_matrix_wanted[1, 0] = np.array((0, 0, 1))
        conv_r_matrix_wanted[1, 2] = np.array((0, 0, 1))
        conv_r_matrix_wanted[1, 3] = np.array((0, 0, 1))
        conv_r_matrix_wanted[2, 1] = np.array((0, 0, 1))
        conv_r_matrix_wanted[2, 3] = np.array((0, 0, 1))
        conv_r_matrix_wanted[3, 1] = np.array((0, 0, 1))
        conv_r_matrix_wanted[3, 2] = np.array((0, 0, 1))
        conv_r_matrix_wanted[3, 3] = np.array((0, 0, 1))
        conv_r_matrix_wanted[4, 0] = np.array((0, 0, 1))
        conv_r_matrix_wanted[4, 2] = np.array((0, 0, 1))
        conv_r_matrix_wanted[4, 3] = np.array((0, 0, 1))
        self.assertTrue(np.all([np.where(~np.isnan(conv_r_matrix))[i] == np.where(~np.isnan(conv_r_matrix_wanted))[i] for i in range(len(np.where(~np.isnan(conv_r_matrix_wanted))))]))

        apriori1 = ard.AssociationRules(conv_r_matrix, 100, curves.Names, 0.3, 0.6)
        freq1, count1 = apriori1.main()
        freq_wanted = [[[[0, 2]], [[1, 2]], [[2, 2]], [[3, 2]]], [[[0, 2], [2, 2]], [[0, 2], [3, 2]], [[1, 2], [2, 2]], [[1, 2], [3, 2]], [[2, 2], [3, 2]]], [[[0, 2], [2, 2], [3, 2]]]]
        sup_wanted = [[3, 3, 4, 4], [3, 2, 2, 2, 3], [2]]
        for i in range(len(freq_wanted)):
            for j in range(len(freq_wanted[i])):
                self.assertEqual(sup_wanted[i][j], count1[i][j])
                for k in range(len(freq_wanted[i][j])):
                    for l in range(len(freq_wanted[i][j][k])):
                        self.assertEqual(freq_wanted[i][j][k][l], freq1[i][j,k,l])


        rules = apriori1.algorithm_main()
        rules_wanted = [np.array([[[0, 2]],
                       [[2, 2]]], dtype=object), np.array([[[2, 2]],
                       [[0, 2]]], dtype=object), np.array([[[0, 2]],
                       [[3, 2]]], dtype=object), np.array([[[1, 2]],
                       [[2, 2]]], dtype=object), np.array([[[1, 2]],
                       [[3, 2]]], dtype=object), np.array([[[2, 2]],
                       [[3, 2]]], dtype=object), np.array([[[3, 2]],
                       [[2, 2]]], dtype=object), np.array([np.array([[0, 2],
                      [2, 2]]), np.array([[3, 2]])], dtype=object), np.array([np.array([[0, 2],
                      [3, 2]]), np.array([[2, 2]])], dtype=object), np.array([np.array([[2, 2],
                      [3, 2]]), np.array([[0, 2]])], dtype=object), np.array([np.array([[0, 2]]), np.array([[2, 2],
                                       [3, 2]])], dtype=object)]
        for i in range(len(rules_wanted)):
            for j in range(len(rules_wanted[i])):
                for k in range(len(rules_wanted[i][j])):
                    for l in range(len(rules_wanted[i][j][k])):
                        self.assertEqual(rules_wanted[i][j][k,l], rules[i][j][k,l])

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

    def test_whole1(self):
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
                                [np.array([1], dtype=object), np.array([2], dtype=object)]], # 0 -> 1 and 2
                                  # [np.array([2], dtype=object), np.array([2], dtype=object)]], # 0 and 1 -> 2 and 2
                                 0.5))



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


    def test_whole2(self):
        curves = fc.Curves1(1, 5, 0.2, 0.45, 0.55, 0.8)
        conv_r_matrix = np.empty((5, 4, 3))
        conv_r_matrix[:] = np.nan
        conv_r_matrix[0, 0] = np.array((0, 0, 1))
        conv_r_matrix[0, 1] = np.array((0, 0, 1))
        conv_r_matrix[0, 2] = np.array((0, 0, 1))
        conv_r_matrix[1, 0] = np.array((0, 0, 1))
        conv_r_matrix[1, 2] = np.array((0, 0, 1))
        conv_r_matrix[1, 3] = np.array((0, 0, 1))
        conv_r_matrix[2, 1] = np.array((0, 0, 1))
        conv_r_matrix[2, 3] = np.array((0, 0, 1))
        conv_r_matrix[3, 1] = np.array((0, 0, 1))
        conv_r_matrix[3, 2] = np.array((0, 0, 1))
        conv_r_matrix[3, 3] = np.array((0, 0, 1))
        conv_r_matrix[4, 0] = np.array((0, 0, 1))
        conv_r_matrix[4, 2] = np.array((0, 0, 1))
        conv_r_matrix[4, 3] = np.array((0, 0, 1))
        recomm = re.Recommend(conv_r_matrix)
        recomm_score = recomm.main_recommend(100, curves.Names, test_size=0.3, cross_num=4, min_support=0.3, min_confidence=0.6)
        recommendations_wanted = [[np.array([0], dtype=object), np.array([0])], [np.array([0], dtype=object), np.array([0])], [], [np.array([2], dtype=object), np.array([2])], [np.array([0], dtype=object), np.array([0])], []]
        score_wanted = 0.5
        self.assertEqual(recomm_score, (recommendations_wanted, score_wanted))

if __name__ == '__main__':
    unittest.main()
