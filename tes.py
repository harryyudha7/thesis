# list_predictor = ['modulo','sepallength','transf1','min','standard']
# data_header = ['sepallength','petalwidth','sepalwidth']
# dict_preprocessing = {'modulo': {'type': 'transform',
#                                  'dependence': ['integer']},
#                     'transf1': {'type': 'transform',
#                                  'dependence': ['petalwidth']},
#                     'integer': {'type': 'transform',
#                                  'dependence': ['sepallength']},
#                     'min': {'type': 'transform',
#                                  'dependence': ['sepallength','petalwidth']},
#                     'standard': {'type': 'standardization',
#                                  'dependence': ['sepalwidth']}
#                     }

# new_header = []
# for i in list_predictor:
#     # col = list_predictor[i]
#     col = i
#     idx = list_predictor.index(i)
#     if col in data_header:
#         if col not in new_header:
#             new_header.append(col)
#     else:
#         if col in dict_preprocessing.keys():
#             for j in dict_preprocessing[col]['dependence']:
#                 if j not in new_header:
#                     new_header.append(j)
#                 if j not in list_predictor:
#                     list_predictor.insert(idx+1,j)
#             if col not in new_header:
#                 new_header.append(col)

# print(new_header)

import re

test_str = "[aaa2]-[PetalWidthCm]*sin([SepalWidthCm])"
 
# printing original string
print("The original string is : " + test_str)
 
# Extract substrings between brackets
# Using regex
res = re.findall(r'\[.*?\]', test_str)
for i in range(len(res)):
    res[i] = res[i].replace('[','').replace(']','')
print(res)