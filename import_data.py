import numpy as np
import pandas as pd
import time
import datetime



class ImportData:
    def __init__(self, dataset):
        self.dataset = dataset
        self.r_matrix = []  # R matrix (users x products, elements: ratings)
        self.t_matrix = []  # T matrix (users x products, elements: timestamps)
        self.users = np.array([], dtype=str)  # already searched users
        self.products = np.array([], dtype=str)  # already searched products
        self.min_score = 0
        self.max_score = 0

    class Parameters:
        def __init__(self, filename, min_score, max_score, user_column, product_column, score_column,
                     time_column=None, parse_date=0, read_time=True, args=None):
            self.filename = filename
            self.min_score = min_score
            self.max_score = max_score
            self.user_column = user_column
            self.product_column = product_column
            self.score_column = score_column
            self.time_column = time_column
            self.parse_date = parse_date
            self.read_time = read_time
            self.args = args

    params_dict = {
        'fine_food': Parameters('Datasets/AmazonFineFoodShort.csv',
                                       1, 5, 'UserId', 'ProductId', 'Score', 'Time'),
        'beauty': Parameters('Datasets/RatingBeautyShort.csv', 1, 5, 'UserId', 'ProductId', 'Rating',
                                    'Timestamp'), # najczęstszy support: 0.0053...
        'products': Parameters('Datasets/AmazonProductsShort.csv',
                                      1, 5, 'reviews.username', 'id', 'reviews.rating',
                                      'reviews.dateAdded', 1),  # ten zbiór nie ma sensu skrócony (abo wgl)
        'products_types': Parameters('Datasets/AmazonProducts.csv',
                                       1, 5, 'reviews.username', 'categories', 'reviews.rating',
                                       'reviews.dateAdded', 1),
        'electronics': Parameters('Datasets/ElectronicsShort3.csv', 1, 5, 0, 1, 2, 3),
        'movies_basic': Parameters('Datasets/MoviesShort.csv', 0.5, 5, 'userId', 'movieId', 'rating', 'timestamp'),
        'smoker': Parameters('Datasets/smokerdata.csv', 1, 5, 'User', 'Brand', 'Rating', read_time=False),
        'movies_short': Parameters('Datasets/movies/ml-latest-small', 0.5, 5, 'userId', 'movieId', 'rating', 'timestamp', read_time=False),
    }

    def import_data(self):
        params = self.params_dict[self.dataset]
        # create database
        if params.read_time:
            df = pd.read_csv(params.filename,
                             usecols=[params.user_column, params.product_column, params.score_column, params.time_column])
        else:
            df = pd.read_csv(params.filename,
                             usecols=[params.user_column, params.product_column, params.score_column])
        self.min_score = params.min_score
        self.max_score = params.max_score
        # initialize lists
        user_max_idx = -1
        user_max_idx = -1
        r_matrix = []
        t_matrix = []
        users = []
        products = []
        # Write to lists
        for index, row in df.iterrows():
            user = row[params.user_column]
            product = row[params.product_column]
            if user not in users:  # to avoid repetition # jakieś bardziej efektywne przeszukiwanie?
                user_max_idx += 1
                users.append(user)
                r_matrix.append([np.nan] * (user_max_idx + 1))
                # if params.read_time:
                #     t_matrix.append([np.nan] * (user_max_idx + 1))
            if product not in products:  # to avoid repetition
                user_max_idx += 1
                products.append(product)
                for i in range(len(r_matrix)):
                    r_matrix[i].append(np.nan)
                    # if params.read_time:
                    #     t_matrix[i].append(np.nan)
                r_matrix[user_max_idx][user_max_idx] = row[params.score_column]
            else:
                find_prod = products.index(product)
                r_matrix[user_max_idx][find_prod] = row[params.score_column]

            # if params.read_time:
            #     # parse timestamps
            #     time_val = row[params.time_column]
            #     # %d%m%y
            #     if params.parse_date == 1:
            #         time_val = time.mktime(datetime.datetime.strptime(time_val, "%Y-%m-%dT%H:%M:%SZ").timetuple())
            #     time_val = float(time_val)
            #     t_matrix[user_max_idx] = time_val



        self.users = np.array(users)
        self.products = np.array(products)
        self.r_matrix = np.array(r_matrix, dtype=object)
        # if params.read_time:
        #     self.t_matrix = np.array(t_matrix)
        print('u')


    def import_movies_genres(self):
        params = self.params_dict[self.dataset]
        # create helper database
        types = pd.read_csv(params.filename + '/movies.csv', usecols=['movieId', 'genres'])
        # create database
        if params.read_time:
            df = pd.read_csv(params.filename,
                             usecols=[params.user_column, params.product_column, params.score_column, params.time_column])
        else:
            df = pd.read_csv(params.filename + '/ratings.csv',
                             usecols=[params.user_column, params.product_column, params.score_column])
        self.min_score = params.min_score
        self.max_score = params.max_score
        # initialize lists
        user_max_idx = -1
        genre_max_idx = -1
        r_matrix = []
        t_matrix = []
        users = []
        genres = []
        # helper matrices
        movie_id_to_genres = {}
        for index, row in types.iterrows():
            movie_id = row['movieId']
            genres_row = row['genres']
            genres_list = genres_row.split('|')
            movie_id_to_genres[movie_id] = genres_list

        # Write to lists
        for index, row in df.iterrows():
            user = row[params.user_column]
            product = row[params.product_column]
            if user not in users:
                user_max_idx += 1
                users.append(user)
                r_matrix.append([[]] * (genre_max_idx + 1))
            for genre in movie_id_to_genres[product]:
                if genre not in genres:  # to avoid repetition
                    genre_max_idx += 1
                    genres.append(genre)
                    for i in range(len(r_matrix)):
                        r_matrix[i].append([])
                    u = r_matrix[user_max_idx][genre_max_idx]
                    r_matrix[user_max_idx][genre_max_idx].append(row[params.score_column])
                else:
                    find_genre = genres.index(genre)
                    r_matrix[user_max_idx][find_genre].append(row[params.score_column])



        self.users = np.array(users)
        self.products = np.array(genres)
        self.r_matrix = np.array(r_matrix, dtype=object)
        # if params.read_time:
        #     self.t_matrix = np.array(t_matrix)
        print('u')