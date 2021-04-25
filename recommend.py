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

    # # split to train and test sets randomly
    # def test_split_random(self, test_size):
    #     # initialize
    #     train_r_matrix = np.empty(self.conv_r_matrix.shape)
    #     train_r_matrix[:] = np.nan
    #     test_r_matrix = np.empty(self.conv_r_matrix.shape)
    #     test_r_matrix[:] = np.nan
    #     train_mask = np.zeros(self.conv_r_matrix.shape, dtype=bool)
    #     test_mask = np.zeros(self.conv_r_matrix.shape, dtype=bool)
    #     for user_idx in range(len(self.conv_r_matrix)):
    #         # products bought by user
    #         user_prod_idxs = np.nonzero(~np.isnan(self.conv_r_matrix[user_idx][:, 0]))
    #         # randomly split products and add them to respective sets
    #         if len(user_prod_idxs[0]) > 1:
    #             train_tmp, test_tmp = train_test_split(user_prod_idxs[0], test_size=test_size)
    #             train_mask[user_idx, train_tmp, :] = True
    #             test_mask[user_idx, test_tmp, :] = True
    #         else:
    #             # if there is only one element
    #             train_mask[user_idx, user_prod_idxs, :] = True
    #
    #     train_r_matrix[train_mask] = self.conv_r_matrix[train_mask]
    #     test_r_matrix[test_mask] = self.conv_r_matrix[test_mask]
    #     return train_r_matrix, test_r_matrix
    #
    # # split to train and test sets by timestamps
    # def test_split_time(self, t_matrix, test_size):
    #     # initialize
    #     train_r_matrix = np.empty(self.conv_r_matrix.shape)
    #     train_r_matrix[:] = np.nan
    #     test_r_matrix = np.empty(self.conv_r_matrix.shape)
    #     test_r_matrix[:] = np.nan
    #     train_mask = np.zeros((self.conv_r_matrix.shape[0], self.conv_r_matrix.shape[1]), dtype=bool)
    #     test_mask = np.zeros((self.conv_r_matrix.shape[0], self.conv_r_matrix.shape[1]), dtype=bool)
    #     # for every user
    #     for user_idx in range(len(self.conv_r_matrix)):
    #         # find where timstemps are not nan
    #         t_not_nan_idx = np.nonzero(~np.isnan(t_matrix[user_idx]))
    #         [t_not_nan] = t_matrix[user_idx, t_not_nan_idx]
    #         # calculate test_size portion of users' time
    #         timestamp_max = max(t_not_nan)
    #         timestamp_min = min(t_not_nan)
    #         diff = timestamp_max - timestamp_min
    #         timestamp_test = timestamp_max - (test_size * diff)
    #         # check what products were bought before timestamp_test
    #         train_mask[user_idx] = t_matrix[user_idx] <= timestamp_test
    #         test_mask[user_idx] = t_matrix[user_idx] > timestamp_test
    #         pass
    #     train_r_matrix[train_mask] = self.conv_r_matrix[train_mask]
    #     test_r_matrix[test_mask] = self.conv_r_matrix[test_mask]
    #     return train_r_matrix, test_r_matrix

    # create and validate recommendations
    def main_recommend(self, S, curves_names, test_size=0.3, cross_num=10, min_support=0.0052, min_confidence=0.9, shuffle_test=False):
        # initialize
        # train_idx = math.floor((1 - test_size) * len(self.conv_r_matrix))
        train, test = train_test_split(self.conv_r_matrix, test_size=test_size, shuffle=shuffle_test) # self.conv_r_matrix[:train_idx], self.conv_r_matrix[train_idx:] # self.test_split_time(t_matrix, test_size)
        apriori = ard.AssociationRules(train, S, curves_names, min_support, min_confidence)
        rules = apriori.algorithm_main()
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
                        # all recommendations for a user precision
                        if len(recomm_in_test_local) > 0:
                            recomm_in_test += mean(recomm_in_test_local)
                # calculate recommendations' precision
                if len(recommendations) > 0:
                    precision = recomm_in_test / len(recommendations)
                    precision_all.append(precision)

        # calculate collective precision
        if len(precision_all) > 0:
            return recommendations_all, mean(precision_all)
        return recommendations_all, 0
