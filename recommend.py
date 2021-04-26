from statistics import mean

import math
import numpy as np
from sklearn.model_selection import train_test_split
import association_rules_division as ard


class Recommend:
    def __init__(self, conv_r_matrix):
        self.conv_r_matrix = conv_r_matrix
        self.rules = []

    def recommend_to_user(self, rules, matrix, user_idx):
        user_p_s_idxs = np.nonzero(~np.isnan(matrix[user_idx]))
        user_p_s = list(zip(user_p_s_idxs[0], user_p_s_idxs[1]))

        recomm_list = []
        for rule in rules:
            antec, conseq = rule
            # check if user bought antequant's elements
            antec_in_user = True
            for elem in antec:
                if tuple(elem) not in user_p_s:
                    antec_in_user = False
                elif matrix[user_idx][elem[0]][elem[1]] <= 0.5:
                    antec_in_user = False

            # check if user bought consequent's elements
            conseq_to_delete = []
            for elem_idx in range(len(conseq)):
                if tuple(conseq[elem_idx]) in user_p_s:
                    conseq_to_delete.append(elem_idx)
            conseq = np.delete(conseq, conseq_to_delete, axis=0)

            # join both conditions
            if antec_in_user and len(conseq) != 0:
                recomm_list.append(conseq[:, 0])
        return recomm_list


    # create and validate recommendations
    def main_recommend(self, S, curves_names, test_size=0.3, cross_num=10, min_support=0.0052, min_confidence=0.9, shuffle_test=False):
        # initialize
        # train_idx = math.floor((1 - test_size) * len(self.conv_r_matrix))
        train, test = train_test_split(self.conv_r_matrix, test_size=test_size, shuffle=shuffle_test) # self.conv_r_matrix[:train_idx], self.conv_r_matrix[train_idx:] # self.test_split_time(t_matrix, test_size)
        apriori = ard.AssociationRules(train, S, curves_names, min_support, min_confidence)
        rules = apriori.algorithm_main()
        print('Rules found')
        self.rules = rules
        # all recommendations
        recommendations_all = []
        precision_all = []


        # cross-validate recommendations on test matrix
        for i_cross in range(cross_num):
            # initialize cross validation cross base and cross test
            cross_base = np.empty(test.shape)
            cross_base[:] = np.nan
            cross_test = np.empty(test.shape)
            cross_test[:] = np.nan

            cross_part_size = math.floor(train.shape[1] / cross_num)
            if i_cross < cross_num-1:
                cross_test_range = range((i_cross * cross_part_size), ((i_cross + 1) * cross_part_size))
                cross_test[:, cross_test_range] = test[:, cross_test_range]
                cross_base[:, :(i_cross * cross_part_size)] = test[:, :(i_cross * cross_part_size)]
                cross_base[:, ((i_cross + 1) * cross_part_size):] = test[:, ((i_cross + 1) * cross_part_size):]
            else:
                cross_test[:, (i_cross * cross_part_size):] = test[:, (i_cross * cross_part_size):]
                cross_base[:, :(i_cross * cross_part_size)] = test[:, :(i_cross * cross_part_size)]


                # recommend products to every user and check if they have really bought it
            for test_user_idx in range(len(test)):
                # recommendations based on cross base
                recommendations = self.recommend_to_user(rules, cross_base, test_user_idx)
                recommendations_all.append(recommendations)
                # other products bought by user (cross test)
                cross_test_p_s_idxs = np.nonzero(~np.isnan(cross_test[test_user_idx]))

                #debug
                debug_r_matrix_p_s_idxs = np.nonzero(~np.isnan(self.conv_r_matrix[test_user_idx]))
                debug_user_cross_test = cross_test[test_user_idx]
                debug_user_test = test[test_user_idx]
                debug_user_r_matrix = self.conv_r_matrix[test_user_idx]

                # count precise recommendations
                recomm_in_test = 0
                # if user bought any products belonging to cross test:
                if len(cross_test_p_s_idxs[0]) > 0:
                    # calculate precision of recommendation of every product
                    for recommendation in recommendations:
                        recomm_in_test_local = []
                        for prod in recommendation:
                            # if product is in cross_test, add its fuzzy function for HIGH to recommendation counter
                            if prod in cross_test_p_s_idxs[0]:
                                recomm_in_test_local.append(self.conv_r_matrix[test_user_idx, prod, len(curves_names) - 1])
                                pass
                        # all recommendations for a user precision
                        if len(recomm_in_test_local) > 0:
                            recomm_in_test += mean(recomm_in_test_local)
                            pass
                # calculate recommendations' precision
                if len(recommendations) > 0:
                    precision = recomm_in_test / len(recommendations)
                    precision_all.append(precision)
                    pass

        # calculate collective precision
        if len(precision_all) > 0:
            return recommendations_all, mean(precision_all)
        return recommendations_all, 0
