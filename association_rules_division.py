import numpy as np
import random
from itertools import combinations, chain, permutations
from scipy.special import comb
import fuzzy_curves as fc
import apriori as apr
import math

class AssociationRules:
    def __init__(self, conv_r_matrix, div_percentage, sets_enum: fc.FuzzyCurves.Names, min_support=0.00000001, min_confidence=0.6):
        self.conv_r_matrix = conv_r_matrix
        self.number_of_transactions = np.count_nonzero(~np.isnan(conv_r_matrix)) / len(conv_r_matrix[0][0])
        self.number_of_users = len(conv_r_matrix)
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.sets_enum = sets_enum
        self.max_div_size = div_percentage * self.number_of_users / 100  # users instead of transactions
        self.r_divisions = []

    # "Product" representation: [index of Product, index of Fuzzy Set]
    ProductScore = np.array([int, int])

    # find maximum level of division and number of divisions
    def division_params(self):
        acc_size = self.number_of_users
        level = 0
        number_of_divisions = 1
        # divide until maximum division size is reached
        while acc_size > self.max_div_size:
            acc_size /= 2
            level += 1
            number_of_divisions *= 2
        return level, number_of_divisions

    # Unite two sub-databases
    def unite(self, min_support, sub_tdb1, sub_tdb2, supports_tdb1, supports_tdb2):
        # sets' lengths in sub_tdb1
        lengths1 = np.array([np.shape(elem)[1] for elem in sub_tdb1])
        ### merge both sub-databases
        for length_idx2 in range(len(sub_tdb2)):
            # iterate through freguent sets in sub_tdb2
            for t_idx in range(len(sub_tdb2[length_idx2])):
                # operate on same sets' lengths in both sub-databases
                length_idx1 = np.where(lengths1 == length_idx2 + 1)
                # if matching length in sub_tdb1 found
                if len(length_idx1[0]) > 0:
                    # check if frequent set already exists in sub_tdb1
                    t_in_tdb1 = np.where(np.all(np.all(sub_tdb2[length_idx2][t_idx] == sub_tdb1[length_idx1[0][0]], axis=1), axis=1))
                    # if set already exists in sub_tdb1, increase its support
                    if len(t_in_tdb1[0]) > 0:
                        supports_tdb1[length_idx2][t_in_tdb1] += supports_tdb2[length_idx2][t_idx]
                    # if not, add it to respective length
                    elif supports_tdb2[length_idx2][t_idx] >= min_support:
                        sub_tdb1[length_idx2] = np.append(sub_tdb1[length_idx2], [sub_tdb2[length_idx2][t_idx]], axis=0)
                        supports_tdb1[length_idx2] = np.append(supports_tdb1[length_idx2], [supports_tdb2[length_idx2][t_idx]], axis=0)
                # if matching length not found, add new row to sub_tdb1
                elif supports_tdb2[length_idx2][t_idx] >= min_support:
                    sub_tdb1.append(np.array(np.expand_dims(sub_tdb2[length_idx2][t_idx], axis=0), dtype=object))
                    supports_tdb1.append(np.array([supports_tdb2[length_idx2][t_idx]]))
                    lengths1 = np.append(lengths1, length_idx2+1)

        # check if updated supports match the condition
        # sup_mask = np.array(supports_tdb1) >= min_support
        for length_idx1 in range(len(sub_tdb1)):
            sup_mask = supports_tdb1[length_idx1] >= min_support
            sub_tdb1[length_idx1] = sub_tdb1[length_idx1][sup_mask]
            supports_tdb1[length_idx1] = supports_tdb1[length_idx1][sup_mask]

        return sub_tdb1, supports_tdb1


    def main(self):
        number_of_levels, number_of_divisions = self.division_params()
        # divide transactional database into parts
        self.r_divisions = np.array_split(self.conv_r_matrix, number_of_divisions)
        frequent_matrix = []
        supports_matrix = []

        # find frequent sets in every division
        for division in self.r_divisions:
            apriori = apr.Apriori(division, self.sets_enum, self.min_support)
            frequent_sets, supports = apriori.apriori()
            frequent_matrix.append(frequent_sets)
            # supports counted in relation to the whole database
            supports_matrix.append([elem / number_of_divisions for elem in supports])

        # UNITING
        # minimum support needed increases with every level
        curr_min_support = self.min_support / number_of_divisions
        for lvl in range(number_of_levels, -1, -1):
            divs = int(math.pow(2, lvl))
            for j in range(int(divs / 2)):
                frequent_matrix[j], supports_matrix[j] = self.unite(curr_min_support, frequent_matrix[j*2], frequent_matrix[j*2+1], supports_matrix[j*2], supports_matrix[j*2+1])
            curr_min_support *= 2
        return frequent_matrix[0], supports_matrix[0]


    def confidence(self, itemset_support, pred_support):
        conf_np = itemset_support / pred_support
        return conf_np


    # Generate association rules based on frequent itemsets
    def generate_rules(self, frequent_sets, supports):
        rules2 = []
        # for all frequent itemsets
        for itemset_length_idx in range(1, len(frequent_sets)):
            itemset_length = itemset_length_idx + 1
            for itemset_idx in range(len(frequent_sets[itemset_length_idx])):
                itemset = frequent_sets[itemset_length_idx][itemset_idx]
                itemset_support = supports[itemset_length_idx][itemset_idx]
                # generate all possible rules from a set
                for pred_length in range(1, itemset_length):
                    for pred_idx in combinations(range(itemset_length), pred_length):
                        # get predecessors' items and support
                        predecessor = itemset[list(tuple(pred_idx))]
                        pred_support_idx = np.nonzero(np.all(np.all(frequent_sets[pred_length-1] == predecessor, axis=1), axis=1))
                        pred_support = supports[pred_length-1][pred_support_idx]

                        successor = np.delete(itemset, pred_idx, 0)
                        # check the confidence
                        conf = self.confidence(itemset_support, pred_support)
                        if conf >= self.min_confidence:
                            #generate rule
                            rules2.append([predecessor, successor])
        return rules2


    def algorithm_main(self):
        frequent_sets, supports = self.main()
        debug = self.generate_rules(frequent_sets, supports)
        return debug






