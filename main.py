import import_data
import apriori as apr
import association_rules_division as ard
import association_rules_pure_python as aprpp
from additional_functions import create_converted_r_matrix
import fuzzy_curves as fc
import recommend as re
import visualizations as vs
import numpy as np

# read data
data = import_data.ImportData('beauty')
data.import_data()

#choose fuzzy curves class
curves = fc.Curves1(data.min_score, data.max_score, 0.2, 0.45, 0.55, 0.8)
#
# # create fuzzy association rules
conv_r_matrix = create_converted_r_matrix(data.r_matrix, curves)
t_matrix = data.t_matrix

# create and validate recommendations
recomm = re.Recommend(conv_r_matrix)
recommendations, recomm_score = recomm.main_recommend(t_matrix, 20, curves.Names, test_size=0.3, min_support=0.000001, min_confidence=0.004)
