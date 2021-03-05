import numpy as np
import pandas as pd

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
        df = pd.read_csv('Datasets/AmazonFineFoodShort3.csv')
        df = df.drop(['Id', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Text', 'Summary'],
                     axis=1)
        # inicjalizacja list
        user_idx = -1
        prod_idx = -1
        # score = df['Score'][0]
        r_matrix = []
        t_matrix = []
        users = []
        products = []

        # rozbudowa list
        for index, row in df.iterrows():
            user = row['UserId'] # nie przypisywać do zmiennej?
            product = row['ProductId']

            if user not in users:  # jakieś bardziej efektywne przeszukiwanie?
                user_idx += 1
                users.append(user)
                r_matrix.append([np.nan] * (prod_idx+1))
                t_matrix.append([np.nan] * (prod_idx+1))

            if product not in products:
                prod_idx += 1
                products.append(product)
                for i in range(len(r_matrix)):
                    r_matrix[i].append(np.nan)
                    t_matrix[i].append(np.nan)

            r_matrix[user_idx][prod_idx] = row['Score']
            t_matrix[user_idx][prod_idx] = row['Time']

        self.users = np.array(users)
        self.products = np.array(products)
        self.r_matrix = np.array(r_matrix, dtype=object)
        self.t_matrix = np.array(t_matrix, dtype=object)
        print('u')



    # konwersje z innych formatów
