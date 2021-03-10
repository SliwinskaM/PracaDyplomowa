import numpy as np
import random
from itertools import combinations

class AssociationRules:
    def __init__(self, conv_r_matrix, min_support=0.00000001, min_confidence=0.6):
        self.conv_r_matrix = conv_r_matrix
        self.number_of_sets = len(conv_r_matrix[0][0])
        self.number_of_transactions = np.count_nonzero(~np.isnan(conv_r_matrix))/self.number_of_sets
        self.number_of_users = len(conv_r_matrix)
        self.number_of_products = len(conv_r_matrix[0])
        self.min_support = min_support
        self.min_confidence = min_confidence

    # "Product" representation: [index of Product, index of Fuzzy Set]
    ProductScore = tuple[int, int]

    def support(self, items: list[ProductScore]):
        count = 0
        for user_transactions_all in self.conv_r_matrix: # for all users
            user_transactions = np.nonzero(~np.isnan(user_transactions_all))[0] # products bought by the user (with repetition)
            items_nums = [item[0] for item in items] # indexes of items
            items_scores = [item[1] for item in items] # idexes of items' scores
            if all(item in user_transactions for item in items_nums): # if all items are in a transaction
                count_temp = []
                for i in range(len(items)):
                    count_temp.append(user_transactions_all[items_nums[i]][items_scores[i]]) # add their score
                count += np.min(count_temp)
        return count / self.number_of_transactions

    # optional
    def support_heuristic(self, items: list[ProductScore]):
        count = 0
        # chosen_sample = random.sample(range(round(self.number_of_users)), round(0.6 * self.number_of_users)) #count only for random sample of users
        # checkpoint = round(0.6 * self.number_of_users)
        count_threshold = self.min_support * self.number_of_transactions # threshold to stop counting after
        # c = 0
        for i in range(len(self.conv_r_matrix)):  # chosen_sample:
            user_transactions_all = self.conv_r_matrix[i]
            user_transactions = np.nonzero(~np.isnan(user_transactions_all))[0] # products bought by the user (with repetition)
            items_nums = [item[0] for item in items] # indexes of items
            items_scores = [item[1] for item in items] # indexes of items' scores
            if all(item in user_transactions for item in items_nums): # if all items are in a transaction
                count_temp = []
                for i in range(len(items)):
                    count_temp.append(user_transactions_all[items_nums[i]][items_scores[i]]) # add their score
                count += np.min(count_temp)
            # heuristics - stop counting when possible
            if count >= count_threshold:
                break
            # if i == checkpoint: # w tym celu trzebaby pomieszać kolejność próbek, dla których są wykonywane obliczenia
            #     if count == 0:
            #         break
            # c += 1
        return count / self.number_of_transactions


    def confidence(self, pred: list[ProductScore], desc: list[ProductScore]):
        # MOŻE: NA PODSTAWIE SUPPORT / HEURYSTYKA
        pred_nums = [item[0] for item in pred]  # indexes of predecessing items
        pred_scores = [item[1] for item in pred]  # idexes of predecessing items' scores
        count_both = 0
        count_pred = 0
        for user_transactions_all in self.conv_r_matrix: # for all users
            user_transactions = np.nonzero(~np.isnan(user_transactions_all))[0] # products bought by the user (with repetition)
            if all(item in user_transactions for item in pred_nums): # if all predecessing items are in a transaction
                count_pred_temp = []
                for i in range(len(pred)):
                    count_pred_temp.append(user_transactions_all[pred_nums[i]][pred_scores[i]]) # add their fuzzy function
                count_pred += np.min(count_pred_temp) # median of the scores
                desc_nums = [item[0] for item in desc]  # indexes of descending items
                desc_scores = [item[1] for item in desc]  # idexes of descending items' scores
                if all(item in user_transactions for item in desc_nums):  # if all descending items are in a transaction
                    count_both_temp = []
                    count_both_temp.append(user_transactions_all[pred_nums[i]][pred_scores[i]]) # add predescessors' fuzzy function
                    for i in range(len(desc)):
                        count_both_temp.append(user_transactions_all[desc_nums[i]][desc_scores[i]])  # add descendors' fuzzy function
                    count_both += np.min(count_both_temp) # median of the scores
        return count_both / count_pred

    # First candidates - all possibilities
    def create_c_1(self):
        c_1 = []
        for product_num in range(self.number_of_products):
            for set_num in range(self.number_of_sets):
                product_score = [product_num, set_num] # join all products and scores (sets' number) possible
                c_1.append([product_score]) # add created products to C1
        return c_1

    # Check candidates' support
    def gen_l_k(self, c_k):
        l_k = []
        for products_scores in c_k:
            if self.support(products_scores) >= self.min_support:
                l_k.append(products_scores) # add accepted products to L
        return l_k

    # Generate candidates
    def gen_c_k(self, l_prev, k):
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
                if k <= 2: # when there are no members < k-2
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

    # Main Apriori
    def apriori(self):
        c = self.create_c_1()
        l = self.gen_l_k(c)
        l_final = []
        for k in range(2, self.number_of_products):
            c = self.gen_c_k(l, k)
            l = self.gen_l_k(c)
            if not l:
                break
            l_final += l
        return l_final

    # Generate association rules based on frequent itemsets
    def generate_rules(self, frequent_sets):
        rules = []
        for itemset in frequent_sets:
            # generate all possible rules from a set
            for pred_length in range(1, len(itemset)):
                for pred in combinations(itemset, pred_length):
                    desc = [item for item in itemset if item not in pred]
                    # check the confidence
                    conf = self.confidence(list(pred), desc)
                    if conf >= self.min_confidence:
                        rules.append([list(pred), desc])
        return rules


    def algorithm_main(self):
        frequent_sets = self.apriori()
        return self.generate_rules(frequent_sets)
