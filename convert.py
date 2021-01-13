import numpy as np
import pandas as pd


# Funkcje przynależności
################ FUZZY CURVES ##################################
def low_curve(score, min_score, max_score):
    # normalize the score to scale 0-1
    score_norm = (score - min_score) / (max_score - min_score)
    if score_norm <= 0.2:
        return 1
    if 0.2 < score_norm < 0.45:
        return round(1 - 4 * (score_norm - 0.2), 2)
    return 0
    # return score_norm


def medium_curve(score, min_score, max_score):
    # normalize the score to scale 0-1
    score_norm = (score - min_score) / (max_score - min_score)
    if 0.2 < score_norm < 0.45:
        return round(4 * (score_norm - 0.2), 2)
    if 0.45 <= score_norm <= 0.55:
        return 1
    if 0.55 < score_norm < 0.8:
        return round(1 - 4 * (score_norm - 0.55), 2)
    return 0


def high_curve(score, min_score, max_score):
    # normalize the score to scale 0-1
    score_norm = (score - min_score) / (max_score - min_score)
    if 0.55 < score_norm < 0.8:
        return round(4 * (score_norm - 0.55), 2)
    if score_norm >= 0.8:
        return 1
    return 0


########### CONVERT ##################
class Convert:
    def __init__(self):
        self.r_matrix = []  # np.array([[[np.nan]]])  # np.full((1, 1, 2), np.nan) ##users x products
        self.t_matrix = []
        self.similarity = []
        self.users = np.array([], dtype=str)
        self.products = np.array([], dtype=str)

    # Import Shortened Amazon Fine Food
    def import_amazon_fine_food_short1(self):
        df = pd.read_csv('Datasets/AmazonFineFoodShort.csv')
        df = df.drop(['Id', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Text', 'Summary'],
                     axis=1)
        r_matrix = [[[np.nan, np.nan, np.nan]]]
        user_idx = 0
        prod_idx = 0
        for index, row in df.iterrows():
            score = row['Score']
            if index == 0:  # ROBOCZO
                r_matrix[user_idx][prod_idx] = [low_curve(score, 1, 5), medium_curve(score, 1, 5), high_curve(score, 1, 5)]
                continue

            if row['UserId'] not in self.users:  # jakieś bardziej efektywne przeszukiwanie?
                user_idx += 1
                self.users = np.append(self.users, row['UserId'])
                r_matrix.append([[np.nan, np.nan, np.nan]] * user_idx)

            if row['ProductId'] not in self.products:
                prod_idx += 1
                self.products = np.append(self.products, row['ProductId'])
                for i in range(len(r_matrix)):
                    r_matrix[i].append([np.nan, np.nan, np.nan])

            r_matrix[user_idx][prod_idx] = [low_curve(score, 1, 5), medium_curve(score, 1, 5), high_curve(score, 1, 5)]

        self.r_matrix = np.array(r_matrix, dtype=object)
        # print('u')
        # print(len(self.users))
        # print(len(self.products))
        # print(len(self.r_matrix))
        # print(self.r_matrix)

    # # Import Shortened Amazon Fine Food REFERENCE PROBLEM
    # def import_amazon_fine_food_short2(self):
    #     df = pd.read_csv('Datasets/AmazonFineFoodShort.csv')
    #     df = df.drop(['Id', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Text', 'Summary'], axis=1)  # może zamiast drop to wybierać?
    #     r_matrix = [[np.nan]*len(df)]*len(df)
    #     user_idx = 0
    #     prod_idx = 0
    #     for index, row in df.iterrows():
    #         if index == 0: #ROBOCZO
    #             print(r_matrix[user_idx][prod_idx])
    #             print(row['Score'])
    #             r_matrix[0][0] = -1
    #             r_matrix[user_idx][prod_idx] = -2
    #             r_matrix[user_idx][prod_idx] = row['Score']
    #             continue
    #
    #         if row['UserId'] not in self.users: #jakieś bardziej efektywne przeszukiwanie?
    #             user_idx += 1
    #             self.users = np.append(self.users, row['UserId'])
    #
    #         if row['ProductId'] not in self.products:
    #             prod_idx += 1
    #             self.products = np.append(self.products, row['ProductId'])
    #
    #         r_matrix[user_idx][prod_idx] = row['Score']
    #
    #     self.r_matrix = np.array(r_matrix, dtype=object)

    # Import Shortened Amazon Fine Food
    def import_amazon_fine_food_short3(self):
        df = pd.read_csv('Datasets/AmazonFineFoodShort.csv')
        df = df.drop(['Id', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Text', 'Summary'],
                     axis=1)  # może zamiast drop to wybierać?
        r_matrix = np.array([[np.nan] * len(df)] * len(df))
        user_idx = 0
        prod_idx = 0
        for index, row in df.iterrows():
            if index == 0:  # ROBOCZO
                r_matrix[user_idx][prod_idx] = row['Score']
                continue

            if row['UserId'] not in self.users:  # jakieś bardziej efektywne przeszukiwanie?
                user_idx += 1
                self.users = np.append(self.users, row['UserId'])

            if row['ProductId'] not in self.products:
                prod_idx += 1
                self.products = np.append(self.products, row['ProductId'])

            r_matrix[user_idx][prod_idx] = row['Score']

        self.r_matrix = r_matrix

    # Import Amazon Fine Food
    def import_amazon_fine_food(self):
        df = pd.read_csv('Datasets/AmazonFineFood.csv')
        df.head()
        df = df.drop(['Id', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Text', 'Summary'],
                     axis=1)  # może zamiast drop to wybierać?
        print(df.head())
        df_R = df.drop([''])  # trzeba to przerobić na wiersze-kolumny
        return df_R.to_numpy()

        # # Import Electronics
        # def import_electronics():
        #     df = pd.read_csv('Datasets/Electronics.csv')
        #     df.head()
        #     print(df.head())
