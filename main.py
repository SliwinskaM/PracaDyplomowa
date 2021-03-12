import matplotlib.pyplot as plt
import numpy as np

import import_data
import association_rules as apr
from additional_functions import create_converted_r_matrix
import fuzzy_curves as fc

# read data
data = import_data.ImportData('movies')
data.import_data()
print('o')
r_matrix = data.r_matrix
t_matrix = data.t_matrix
#choose fuzzy curves class
curves = fc.Curves1(data.min_score, data.max_score)
# create fuzzy association rules
conv_r_matrix = create_converted_r_matrix(r_matrix, curves)
apriori = apr.AssociationRules(conv_r_matrix, curves.Names, 0.0001)
rules = apriori.algorithm_main()
print(rules)


# Wizualizacja funkcji przynależności
# ########### PLOTS #######################
# x = np.linspace(0.0, 1.0)
# sc = np.linspace(1.0, 5.0)
# y1 = [fc.Curves2(dt.min_score, dt.max_score).low_curve(i) for i in sc]
# y3 = [fc.Curves2(dt.min_score, dt.max_score).high_curve(i) for i in sc]
# plt.plot(x, y1)
# plt.plot(x, y3)
# plt.show()
