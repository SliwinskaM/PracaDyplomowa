import numpy as np
import random
from itertools import combinations, chain, permutations
from scipy.special import comb
import fuzzy_curves as fc

class AssociationRules:
    def __init__(self, conv_r_matrix, sets_enum: fc.FuzzyCurves.Names, min_support=0.00000001, min_confidence=0.6):
        self.conv_r_matrix = conv_r_matrix
        self.number_of_sets = len(conv_r_matrix[0][0])
        self.number_of_transactions = np.count_nonzero(~np.isnan(conv_r_matrix))/self.number_of_sets
        self.number_of_users = len(conv_r_matrix)
        self.number_of_products = len(conv_r_matrix[0])
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.sets_enum = sets_enum
        self.transactions_all_np = np.nonzero(~np.isnan(self.conv_r_matrix[:, :, 0]))

    # "Product" representation: [index of Product, index of Fuzzy Set]
    ProductScore = np.array([int, int]) #str]


    def support_numpy(self, items):
        count_np = 0
        items_nums = np.array(items[:, 0]) # indexes of items
        items_scores = np.array(items[:, 1])  # indexes of items' scores
        items_in_transactions_mask = np.isin(self.transactions_all_np[1], items_nums)
        users_for_items = self.transactions_all_np[0][items_in_transactions_mask]
        unique, counts = np.unique(users_for_items, return_counts=True)
        users_ok = unique[counts == len(items)]
        if users_ok.size > 0:
            rows_supports = self.conv_r_matrix[users_ok][:, items_nums, items_scores]
            counts = np.min(rows_supports, axis=1)
            count_np = sum(counts)
        return count_np / self.number_of_transactions



    def confidence(self, pred, desc):
        conf_np = self.support_numpy(np.append(pred, desc, axis=0)) / self.support_numpy(pred)
        return conf_np

    # First candidates - all possibilities
    def create_c_1(self):
        c_1 = np.transpose(np.meshgrid(range(self.number_of_products), self.sets_enum)).reshape(-1, 1, 2)
        return c_1

    # Check candidates' support
    def gen_l_k(self, c_k):
        l_k_mask = np.nonzero([[self.support_numpy(row) >= self.min_support] for row in c_k]) # itemsets with proper support
        l_k = c_k[l_k_mask[0]]
        return l_k




    #Helper function
    def comb_index(self, n, k):
        count = comb(n, k, exact=True)
        index = np.fromiter(chain.from_iterable(combinations(range(n), k)),
                            int, count=count * k)
        return index.reshape(-1, k)

    # Generate candidates
    def gen_c_k(self, l_prev, k):
        # JOINING
        set_pairs_all = np.array(np.meshgrid(range(len(l_prev)), range(len(l_prev)))).T.reshape(-1,2)  # pary indeksów zbiorów do potencjalnego połączenia
        set_pairs = np.unique(np.sort(set_pairs_all,axis=1),axis=0)
        join_1 = [[True]] * len(set_pairs)
        if k > 2:
            # sort arrays to compare
            pair0 = l_prev[set_pairs[:, 0], :k - 2]
            pair1 = l_prev[set_pairs[:, 1], :k - 2]
            join_1 = np.all(np.all(np.equal(pair0, pair1), axis=1), axis=1).reshape(-1,1)
        join_2 = ~(np.equal(l_prev[set_pairs[:,0], k - 2][:,0], l_prev[set_pairs[:,1], k - 2][:,0])).reshape(-1,1)
        join_both = np.concatenate((join_1, join_2), axis=1)
        join_np = np.all(join_both, axis=1) # oba warunki połączneia spełnione
        to_join = set_pairs[join_np] #pary indeksów zbiorów do połączenia

        if np.any(to_join):
            # to retain order - item with bigger index goes first
            first_0 = l_prev[to_join[:,0], k - 2][:,0] < l_prev[to_join[:,1], k - 2][:,0]
            first_0 = np.stack((first_0, first_0)).T
            firsts = np.where(first_0, l_prev[to_join[:,0], k - 2], l_prev[to_join[:,1], k - 2])
            seconds = np.where(~first_0, l_prev[to_join[:,0], k - 2], l_prev[to_join[:,1], k - 2])

            # final joining
            if k <= 2:
                candidates = np.array(list(zip(firsts, seconds)))
            else:
                candidates = np.concatenate((l_prev[to_join[:, 0], :k-2], np.expand_dims(firsts, axis=1), np.expand_dims(seconds, axis=1)), axis=-2)
        else:
            candidates = np.array([])

        # PRUNING
        if np.any(to_join):
            subsets_idx = self.comb_index(k, k-1)
            subsets = candidates[:, subsets_idx]
            prune_mask = np.all(np.all(np.all(np.isin(subsets, l_prev), axis=1), axis=1), axis=1)  #to może nie do końca działać - zrób coś raczej z any
            candidates = candidates[prune_mask]
        candidates = np.unique(candidates, axis=0)
        return candidates




    # Main Apriori
    def apriori(self):
        c = self.create_c_1()
        l = self.gen_l_k(c)
        l_final = [] # np.empty([1, 1, 1], dtype=object)
        for k in range(2, self.number_of_products):
            c = self.gen_c_k(l, k)
            l = self.gen_l_k(c)
            if l.size == 0:
                break
            l_final.append(l)
        return l_final # np.array(l_final) #ulepszyć jakoś?


    # Generate association rules based on frequent itemsets
    def generate_rules(self, frequent_sets):
        rules2 = []
        for itemset_length_idx in range(len(frequent_sets)):
            itemset_length = itemset_length_idx + 2
            for itemset in frequent_sets[itemset_length_idx]:
                # generate all possible rules from a set
                for pred_length in range(1, itemset_length):
                    for pred_idx in combinations(range(itemset_length), pred_length):
                        pred = itemset[list(tuple(pred_idx))]
                        desc = np.delete(itemset, pred_idx, 0)
                        # check the confidence
                        conf = self.confidence(pred, desc)
                        if conf >= self.min_confidence:
                            rules2.append([pred, desc])
        return rules2


    def algorithm_main(self):
        frequent_sets = self.apriori()
        debug = self.generate_rules(frequent_sets)
        return debug
