import numpy as np
import pandas as pd


class ImportData:
    def __init__(self, dataset):
        self.dataset = dataset
        self.r_matrix = []  # R matrix (users x products, elements: ratings)
        self.t_matrix = []  # T matrix (users x products, elements: timestamps)
        self.similarity = []  # optional
        self.users = np.array([], dtype=str)  # already searched users
        self.products = np.array([], dtype=str)  # already searched products
        self.min_score = 0
        self.max_score = 0

    class Parameters:
        def __init__(self, filename, min_score, max_score, user_column, product_column, score_column,
                     time_column, args=None):
            self.filename = filename
            self.min_score = min_score
            self.max_score = max_score
            self.user_column = user_column
            self.product_column = product_column
            self.score_column = score_column
            self.time_column = time_column
            self.args = args

    params_dict = {
        'fine_food': Parameters('Datasets/AmazonFineFoodShort3.csv',
                                       1, 5, 'UserId', 'ProductId', 'Score', 'Time'),
        'beauty': Parameters('Datasets/RatingBeautyShort.csv', 1, 5, 'UserId', 'ProductId', 'Rating',
                                    'Timestamp'), # najczęstszy support: 0.0053...
        'products': Parameters('Datasets/AmazonProductsShort.csv',
                                      1, 5, 'reviews.username', 'id', 'reviews.rating',
                                      'reviews.dateAdded'),  # ten zbiór nie ma sensu skrócony (abo wgl)
        'electronics': Parameters('Datasets/ElectronicsShort.csv', 1, 5, 0, 1, 2, 3),
        'movies': Parameters('Datasets/MoviesShort.csv', 0.5, 5, 'userId', 'movieId', 'rating', 'timestamp') # to 0.5 podejrzane
    }

    def import_data(self):
        params = self.params_dict[self.dataset]
        # create database
        df = pd.read_csv(params.filename,
                         usecols=[params.user_column, params.product_column, params.score_column, params.time_column])
        # df = df.drop(params.columns_to_drop, axis=1)
        self.min_score = params.min_score
        self.max_score = params.max_score
        # initialize lists
        user_idx = -1
        prod_idx = -1
        r_matrix = []
        t_matrix = []
        users = []
        products = []
        # Write to lists
        for index, row in df.iterrows():
            user = row[params.user_column]
            product = row[params.product_column]
            if user not in users:  # to avoid repetition # jakieś bardziej efektywne przeszukiwanie?
                user_idx += 1
                users.append(user)
                r_matrix.append([np.nan] * (prod_idx + 1))
                t_matrix.append([np.nan] * (prod_idx + 1))
            if product not in products:  # to avoid repetition
                prod_idx += 1
                products.append(product)
                for i in range(len(r_matrix)):
                    r_matrix[i].append(np.nan)
                    t_matrix[i].append(np.nan)
            r_matrix[user_idx][prod_idx] = row[params.score_column]
            t_matrix[user_idx][prod_idx] = row[params.time_column]
        self.users = np.array(users)
        self.products = np.array(products)
        self.r_matrix = np.array(r_matrix, dtype=object)
        self.t_matrix = np.array(t_matrix, dtype=object)
        print('u')

