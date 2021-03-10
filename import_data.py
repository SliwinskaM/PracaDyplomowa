import numpy as np
import pandas as pd

class ImportData:
    def __init__(self):
        self.r_matrix = []  # R matrix (users x products, elements: ratings)
        self.t_matrix = []  # T matrix (users x products, elements: timestamps)
        self.similarity = []  # optional
        self.users = np.array([], dtype=str)  # already searched users
        self.products = np.array([], dtype=str)  # already searched products
        self.min_score = 0
        self.max_score = 0

    # Import Shortened Amazon Fine Food
    def import_amazon_fine_food_short(self):
        self.min_score = 1
        self.max_score = 5

        # create database
        df = pd.read_csv('Datasets/AmazonFineFoodShort2.csv')
        df = df.drop(['Id', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Text', 'Summary'],
                     axis=1)
        # initialize lists
        user_idx = -1
        prod_idx = -1
        r_matrix = []
        t_matrix = []
        users = []
        products = []


        # Write to lists
        for index, row in df.iterrows():
            user = row['UserId']
            product = row['ProductId']

            if user not in users: # to avoid repetition # jakieś bardziej efektywne przeszukiwanie?
                user_idx += 1
                users.append(user)
                r_matrix.append([np.nan] * (prod_idx+1))
                t_matrix.append([np.nan] * (prod_idx+1))

            if product not in products: # to avoid repetition
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



    # to do: konwersje z innych formatów
