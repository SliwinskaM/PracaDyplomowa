from time import time
import matplotlib.pyplot as plt
import numpy as np

import convert
import apriori_algorithm as apr
from additional_functions import create_converted_r_matrix


cn = convert.Convert()
# start = time()
cn.import_amazon_fine_food_short()
# end = time()
# print(end - start)
r_matrix = cn.r_matrix
t_matrix = cn.t_matrix
conv_r_matrix = create_converted_r_matrix(r_matrix)
apriori = apr.Apriori(conv_r_matrix, ["Low", "Medium", "High"])
reg = apriori.apriori_main()


# Wizualizacja funkcji przynależności
# ########### PLOTS #######################
# x = np.linspace(0.0, 1.0)
# sc = np.linspace(1.0, 5.0)
# y1 = [convert.low_curve(i, 1, 5) for i in sc]
# y2 = [convert.medium_curve(i, 1, 5) for i in sc]
# y3 = [convert.high_curve(i, 1, 5) for i in sc]
# plt.plot(x, y1)
# plt.plot(x, y2)
# plt.plot(x, y3)
# plt.show()
