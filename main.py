import convert
import apriori_algorithm as apr
from time import time
import matplotlib.pyplot as plt

import numpy as np

"""
macierz R (wiersze users, kolumny products, elementy ratings). To jest podstawowa postać używana np. w algorytmie Korena
macierz T (wiersze users, kolumny products, elementy timestamps). Czas wystawienia oceny (tam gdzie ratings niezerowe)
macierz lub callable określająca podobieństwo produktów. Czyli wymiar product x product lub funkcja K(i,j) która zwróci wartość określającą podobieństwo dla produktów o indeksach (i,j). (Na końcu).
"""
cn = convert.Convert()
# start1 = time()
cn.import_amazon_fine_food_short1()
# end1 = time()
# print(end1 - start1)
r_matrix = cn.r_matrix


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
