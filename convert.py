import numpy as np
import pandas as pd


# Funkcje przynależności
############## FUZZY CURVES ##################
def low_curve(score, min_score, max_score):
    # normalize the score to scale 0-1
    score_norm = (score - min_score) / (max_score - min_score)
    if score_norm <= 0.2:
        return 1
    if 0.2 < score_norm < 0.45:
        return round(1 - 4 * (score_norm - 0.2), 2)
    return 0


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


# Import i rozmycie danych
########### CONVERT ##################
class Convert:
    def __init__(self):
        self.r_matrix = []  # macierz R (wiersze users, kolumny products, elementy ratings - rozmyte)
        self.t_matrix = []  # macierz T (wiersze users, kolumny products, elementy timestamps)
        self.similarity = []  # opcjonalnie - macierz podobieństwa
        self.users = np.array([], dtype=str)  # nazwy przeszukanych użytkowników
        self.products = np.array([], dtype=str)  # nazwy przeszukanych produktów


    # Import Shortened Amazon Fine Food
    def import_amazon_fine_food_short(self):
        # utworzenie bazy danych
        df = pd.read_csv('Datasets/AmazonFineFoodShort.csv')
        df = df.drop(['Id', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Text', 'Summary'],
                     axis=1)
        # inicjalizacja list
        user_idx = 0
        prod_idx = 0
        score = df['Score'][0]
        r_matrix = [[[low_curve(score, 1, 5), medium_curve(score, 1, 5), high_curve(score, 1, 5)]]]
        t_matrix = [[df['Time'][0]]]
        users = [df['UserId'][0]]
        products = [df['ProductId'][0]]

        # rozbudowa list
        for index, row in df.iloc[1:].iterrows():
            score = row['Score']
            user = row['UserId']
            product = row['ProductId']

            if user not in users:
                user_idx += 1
                users.append(user)
                r_matrix.append([[np.nan, np.nan, np.nan]] * (prod_idx+1))
                t_matrix.append([np.nan] * (prod_idx+1))

            if product not in products:
                prod_idx += 1
                products.append(product)
                for i in range(len(r_matrix)):
                    r_matrix[i].append([np.nan, np.nan, np.nan])
                    t_matrix[i].append(np.nan)

            r_matrix[user_idx][prod_idx] = [low_curve(score, 1, 5), medium_curve(score, 1, 5), high_curve(score, 1, 5)]
            t_matrix[user_idx][prod_idx] = row['Time']

        self.users = np.array(users)
        self.products = np.array(products)
        self.r_matrix = np.array(r_matrix, dtype=object)
        self.t_matrix = np.array(t_matrix, dtype=object)
