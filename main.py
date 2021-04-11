import import_data
import association_rules_division as ard
from additional_functions import create_converted_r_matrix
import fuzzy_curves as fc

# read data
data = import_data.ImportData('beauty')
data.import_data()
r_matrix = data.r_matrix
#choose fuzzy curves class
curves = fc.Curves1(data.min_score, data.max_score)
# convert R matrix to fuzzy numbers
conv_r_matrix = create_converted_r_matrix(r_matrix, curves)
# find association rules
asso_rules = ard.AssociationRules(conv_r_matrix, 20, curves.Names, 0.0052)
rules = asso_rules.algorithm_main()
print(rules)

