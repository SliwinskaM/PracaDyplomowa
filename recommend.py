from statistics import mean

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

    # split to train and test sets randomly
    def test_split_random(self, test_size):
        # initialize
        train_r_matrix = np.empty(self.conv_r_matrix.shape)
        train_r_matrix[:] = np.nan
        test_r_matrix = np.empty(self.conv_r_matrix.shape)
        test_r_matrix[:] = np.nan
        train_mask = np.zeros(self.conv_r_matrix.shape, dtype=bool)
        test_mask = np.zeros(self.conv_r_matrix.shape, dtype=bool)
        for user_idx in range(len(self.conv_r_matrix)):
            # products bought by user
            user_prod_idxs = np.nonzero(~np.isnan(self.conv_r_matrix[user_idx][:, 0]))
            # randomly split products and add them to respective sets
            if len(user_prod_idxs[0]) > 1:
                train_tmp, test_tmp = train_test_split(user_prod_idxs[0], test_size=test_size)
                train_mask[user_idx, train_tmp, :] = True
                test_mask[user_idx, test_tmp, :] = True
            else:
                # if there is only one element
                train_mask[user_idx, user_prod_idxs, :] = True

        train_r_matrix[train_mask] = self.conv_r_matrix[train_mask]
        test_r_matrix[test_mask] = self.conv_r_matrix[test_mask]
        return train_r_matrix, test_r_matrix

    # split to train and test sets by timestamps
    def test_split_time(self, t_matrix, test_size):
        # initialize
        train_r_matrix = np.empty(self.conv_r_matrix.shape)
        train_r_matrix[:] = np.nan
        test_r_matrix = np.empty(self.conv_r_matrix.shape)
        test_r_matrix[:] = np.nan
        train_mask = np.zeros((self.conv_r_matrix.shape[0], self.conv_r_matrix.shape[1]), dtype=bool)
        test_mask = np.zeros((self.conv_r_matrix.shape[0], self.conv_r_matrix.shape[1]), dtype=bool)
        # for every user
        for user_idx in range(len(self.conv_r_matrix)):
            # find where timstemps are not nan
            t_not_nan_idx = np.nonzero(~np.isnan(t_matrix[user_idx]))
            [t_not_nan] = t_matrix[user_idx, t_not_nan_idx]
            # calculate test_size portion of users' time
            timestamp_max = max(t_not_nan)
            timestamp_min = min(t_not_nan)
            diff = timestamp_max - timestamp_min
            timestamp_test = timestamp_max - (test_size * diff)
            # check what products were bought before timestamp_test
            train_mask[user_idx] = t_matrix[user_idx] <= timestamp_test
            test_mask[user_idx] = t_matrix[user_idx] > timestamp_test
            pass
        train_r_matrix[train_mask] = self.conv_r_matrix[train_mask]
        test_r_matrix[test_mask] = self.conv_r_matrix[test_mask]
        return train_r_matrix, test_r_matrix

    # create and validate recommendations
    def main_recommend(self, t_matrix, S, curves_names, test_size=0.3, min_support=0.0052, min_confidence=0.9):
        # initialize
        train, test = self.test_split_time(t_matrix, test_size)
        apriori = ard.AssociationRules(train, S, curves_names, min_support, min_confidence)
        rules = apriori.algorithm_main()
        self.rules = rules
        # all recommendations
        recommendations_all = []
        precision_all = []

        # recommend products for every user and check if they really bought it
        for test_user_idx in range(len(test)):
            test_p_s_idxs = np.nonzero(~np.isnan(self.conv_r_matrix[test_user_idx]))
            recomm = self.recommend_to_user(rules, train, test_user_idx)
            recommendations_all.append(recomm)
            recomm_in_test = 0
            for prod in recomm:
                # if user bought the product, add its fuzzy function for HIGH to recommendation counter
                if prod in test_p_s_idxs[0]:
                    recomm_in_test += self.conv_r_matrix[test_user_idx, prod[0], len(curves_names) - 1]
            # calculate recommendations' precision
            if len(recomm) > 0:
                precision = recomm_in_test / len(recomm)
                precision_all.append(precision)

        # calculate collective precision
        if len(precision_all) > 0:
            return recommendations_all, mean(precision_all)
        return recommendations_all, 0
