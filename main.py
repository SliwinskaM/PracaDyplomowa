import matplotlib.pyplot as plt
import numpy as np

import import_data
import apriori as apr
import association_rules_division as ard
import association_rules_pure_python as aprpp
from additional_functions import create_converted_r_matrix
import fuzzy_curves as fc

# read data
data = import_data.ImportData('beauty')
data.import_data()
print('o')
r_matrix = data.r_matrix
t_matrix = data.t_matrix
#choose fuzzy curves class
curves = fc.Curves1(data.min_score, data.max_score)
# create fuzzy association rules
conv_r_matrix = create_converted_r_matrix(r_matrix, curves)

apriori1 = apr.Apriori(conv_r_matrix, curves.Names, 0.0052)
freq1, sup1 = apriori1.apriori()

apriori2 = ard.AssociationRules(conv_r_matrix, 20, curves.Names, 0.0052)
freq2, sup2 = apriori2.main()

apriori3 = aprpp.AssociationRules(conv_r_matrix, curves.Names, 0.0052)
freq3, sup3 = apriori1.apriori()


equal = []
for i in range(len(freq1)):
    u = freq1[i]
    uu = freq3[i]
    uuu = u == uu
    equal.append(np.all(np.all(uuu)))


rules1 = apriori1.algorithm_main()
rules2 = apriori2.algorithm_main()
# equal = []
# for i in range(len(rules1)):
#     u = rules1[i]
#     uu = rules2[i]
#     uuu = u == uu
#     equal.append(np.all(np.all(uuu)))
pass


# Wizualizacja funkcji przynależności
# ########### PLOTS #######################
# x = np.linspace(0.0, 1.0)
# sc = np.linspace(1.0, 5.0)
# y1 = [fc.Curves2(dt.min_score, dt.max_score).low_curve(i) for i in sc]
# y3 = [fc.Curves2(dt.min_score, dt.max_score).high_curve(i) for i in sc]
# plt.plot(x, y1)
# plt.plot(x, y3)
# plt.show()
