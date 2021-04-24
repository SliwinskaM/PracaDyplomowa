import numpy as np
import matplotlib.pyplot as plt
import csv

########## Wizualizacja macierzy ################################
def write_matrix(data, conv_r_matrix, curves):
    with open('matrixR.csv', 'w', newline='') as csvfile:
        r_writer = csv.writer(csvfile, delimiter=',')
        tmp_first = [""]
        for i in range(len(data.products)):
            tmp_first += [str(data.products[i])]
        r_writer.writerow(tmp_first)
        for row in range(len(conv_r_matrix)):
            tmp_row = []
            tmp_row += [str(data.users[row])]
            for rating in conv_r_matrix[row]:
                tmp = [""]
                if np.all(~np.isnan(rating)):
                    for curve in range(len(curves.Names)):
                        tmp[0] += curves.Names(curve).name + ":" + str(round(rating[curve], 2)) + "\n"
                tmp_row += tmp
            r_writer.writerow(tmp_row)


# Wizualizacja funkcji przynależności
# ########### PLOTS #######################
def plot_fuzzy(data, fc, sets_num, p1, p2, p3, p4):
    x = np.linspace(0.0, 1.0, num=1000)
    sc = np.linspace(1.0, 5.0, num=1000)
    # 3 sets
    if sets_num == 3:
        y1 = [fc.Curves1(data.min_score, data.max_score, p1, p2, p3, p4).low_curve(i) for i in sc]
        y2 = [fc.Curves1(data.min_score, data.max_score, p1, p2, p3, p4).medium_curve(i) for i in sc]
        y3 = [fc.Curves1(data.min_score, data.max_score, p1, p2, p3, p4).high_curve(i) for i in sc]
        plt.plot(x, y1, 'mediumblue')
        plt.plot(x, y2, 'g')
        plt.plot(x, y3, 'r')
        for p in p1, p2, p3, p4:
            plt.axvline(p, c='grey', ls="--")
            plt.text(p, 1.07, str(p), horizontalalignment='center')
        plt.legend(['Low', 'Medium', 'High'], loc="center right", bbox_to_anchor=(1.1, 0.5))

    # 2 sets
    if sets_num == 2:
        y4 = [fc.Curves2(data.min_score, data.max_score, p1, p2).low_curve(i) for i in sc]
        y5 = [fc.Curves2(data.min_score, data.max_score, p1, p2).high_curve(i) for i in sc]
        plt.plot(x, y4, 'mediumblue')
        plt.plot(x, y5, 'r')
        for p in p1, p2:
            plt.axvline(p, c='grey', ls="--")
            plt.text(p, 1.07, str(p), horizontalalignment='center')
        plt.legend(['Low', 'High'], loc="center right", bbox_to_anchor=(1.1, 0.5))

    # common part
    plt.text(0.5, 1.15, 'Punkty graniczne', horizontalalignment='center')
    plt.xlabel('Rozmyta ocena')
    plt.ylabel('Przynależność do zbioru')
    plt.savefig('fuzzy_curve_2_pred.png', format="png")
    plt.show()