import pandas as pd
import numpy as np
from math import isnan
from fuzzy_curves import low_curve, medium_curve, high_curve


def create_converted_r_matrix(prev_r_matrix):
    score = prev_r_matrix[0][0]
    conv_r_matrix = np.empty([len(prev_r_matrix), len(prev_r_matrix[0]), 3]) # [[[low_curve(score, 1, 5), medium_curve(score, 1, 5), high_curve(score, 1, 5)]]]
    conv_r_matrix[:] = np.nan

    for user_idx in range(len(conv_r_matrix)):
        for prod_idx in range(len(conv_r_matrix[0])):
            score = prev_r_matrix[user_idx][prod_idx]
            if not isnan(score):
                conv_r_matrix[user_idx][prod_idx] = [low_curve(score, 1, 5), medium_curve(score, 1, 5), high_curve(score, 1, 5)]

    return conv_r_matrix