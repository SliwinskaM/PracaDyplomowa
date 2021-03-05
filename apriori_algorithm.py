import numpy as np
from copy import deepcopy
from itertools import combinations

class Apriori:
    def __init__(self, conv_r_matrix, sets_names, support_threshold=0.0001):
        self.conv_r_matrix = conv_r_matrix
        self.sets_names = sets_names
        self.number_of_transactions = np.count_nonzero(~np.isnan(conv_r_matrix))/3
        self.number_of_products = len(conv_r_matrix[0])
        self.number_of_sets = len(conv_r_matrix[0][0])
        self.support_threshold = support_threshold

    ProductScore = tuple[int, int]

    def support(self, items: list[ProductScore]):
        count = 0
        # TU BĘDZIE HEURYSTYKA
        for user_transactions_all in self.conv_r_matrix: # for all users
            user_transactions = np.nonzero(~np.isnan(user_transactions_all))[0] # products bought by the user (with repetition)
            items_nums = [item[0] for item in items] # indexes of items
            items_scores = [item[1] for item in items] # idexes of items' scores
            if all(item in user_transactions for item in items_nums): # if all items are in a transaction
                for i in range(len(items)):
                    count += user_transactions_all[items_nums[i]][items_scores[i]] # add their fuzzy function
        return count / self.number_of_transactions


    def confidence(self, pred: list[ProductScore], desc: list[ProductScore]):
        # TU PEWNIE TEŻ BĘDZIE HEURYSTYKA
        count_both = 0
        count_pred = 0
        for user_transactions_all in self.conv_r_matrix: # for all users
            user_transactions = np.nonzero(~np.isnan(user_transactions_all))[0] # products bought by the user (with repetition)
            pred_nums = [item[0] for item in pred] # indexes of predecessing items
            pred_scores = [item[1] for item in pred] # idexes of predecessing items' scores
            if all(item in user_transactions for item in pred_nums): # if all predecessing items are in a transaction
                for i in range(len(pred)):
                    count_pred += user_transactions_all[pred_nums[i]][pred_scores[i]] # add their fuzzy function
                desc_nums = [item[0] for item in desc]  # indexes of descending items
                desc_scores = [item[1] for item in desc]  # idexes of descending items' scores
                if all(item in user_transactions for item in desc_nums):  # if all descending items are in a transaction
                    count_both += user_transactions_all[pred_nums[i]][pred_scores[i]] # add predescessors' fuzzy function
                    for i in range(len(desc)):
                        count_both += user_transactions_all[desc_nums[i]][desc_scores[i]]  # add descendors' fuzzy function
        return count_both / count_pred


    def create_C_1(self):
        c_1 = []
        for product_num in range(self.number_of_products):
            for set1 in range(self.number_of_sets):
                product_score = [product_num, set1] # join all products and scores
                c_1.append([product_score]) # add created products to C1
        return c_1


    def gen_L_k(self, c_k):
        l_k = []
        for products_scores in c_k:
            if self.support(products_scores) >= self.support_threshold:
                l_k.append(products_scores) # add accepted products to L
        return l_k


    def gen_C_k(self, l_prev, k):
        # JOINING
        candidates = []
        for set1, set2 in combinations(range(len(l_prev)), 2):
            join = True
            # compare all members except the last
            for i in range(k-2):
                if l_prev[set1][i] == l_prev[set2][i]:
                    join = True #pass?
                else:
                    join = False
                    break
            # compare last members
            if l_prev[set1][k - 2][0] == l_prev[set2][k - 2][0]:
                join = False
            if join:
                # to avoid repetitions and to retain order
                first = min(l_prev[set1][k - 2], l_prev[set2][k - 2])
                second = max(l_prev[set1][k - 2], l_prev[set2][k - 2])
                if k <= 2: # when there are no members < k-2 #jakoś ładniej?
                    candidates.append([first, second])
                    continue
                candidates.append([l_prev[set1][:k - 2], first, second])

        # PRUNING
        for candidate in candidates:
            for subset in combinations(candidate, k-1):
                if list(subset) not in l_prev:
                    candidates.remove(candidate)
                    break
        return candidates


    def apriori_main(self):
        c = self.create_C_1()
        l = self.gen_L_k(c)
        l_final = l
        k = 2
        while l:
            c = self.gen_C_k(l, k)
            l = self.gen_L_k(c)
            l_final.append(l)
            k += 1
        print(l_final)
        print('u')
        return l_final

