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

# import re

# test_str = "[aaa2]-[PetalWidthCm]*sin([SepalWidthCm])"
 
# # printing original string
# print("The original string is : " + test_str)
 
# # Extract substrings between brackets
# # Using regex
# res = re.findall(r'\[.*?\]', test_str)
# for i in range(len(res)):
#     res[i] = res[i].replace('[','').replace(']','')
# print(res)
# list_dict_preprocessing = [
#       {
#          "transf-1":{
#             "type":"transform",
#             "formula":"[SepalLengthCm]+[SepalWidthCm]"
#          }
#       },
#       {
#          "transf-2":{
#             "type":"transform",
#             "formula":"[SepalLengthCm]+[PetalLengthCm]"
#          }
#       },
#       {
#          "xx":{
#             "type":"transform",
#             "formula":"[SepalLengthCm]+[PetalLengthCm]"
#          }
#       },
#       {
#          "yy":{
#             "type":"transform",
#             "formula":"[SepalLengthCm]+[PetalLengthCm]"
#          }
#       },
#       {
#          "zz":{
#             "type":"transform",
#             "formula":"[SepalLengthCm]+[PetalLengthCm]"
#          }
#       },
#       {
#          "teststand":{
#             "type":"Standarization",
#             "formula":"[SepalLengthCm]"
#          }
#       },
#       {
#          "teststandx":{
#             "type":"Standarization",
#             "formula":"[SepalWidthCm]"
#          }
#       },
#       {
#          "teststandxx":{
#             "type":"Normalization",
#             "formula":"[PetalWidthCm]"
#          }
#       }
#    ]
# dict_preprocessing = {}
# for i in list_dict_preprocessing:
#     dict_preprocessing[list(i.keys())[0]] = list(i.values())[0]

# print(dict_preprocessing)
import pandas as pd
a = {"Id":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150],"SepalLengthCm":[5.1,4.9,4.7,4.6,5,5.4,4.6,5,4.4,4.9,5.4,4.8,4.8,4.3,5.8,5.7,5.4,5.1,5.7,5.1,5.4,5.1,4.6,5.1,4.8,5,5,5.2,5.2,4.7,4.8,5.4,5.2,5.5,4.9,5,5.5,4.9,4.4,5.1,5,4.5,4.4,5,5.1,4.8,5.1,4.6,5.3,5,7,6.4,6.9,5.5,6.5,5.7,6.3,4.9,6.6,5.2,5,5.9,6,6.1,5.6,6.7,5.6,5.8,6.2,5.6,5.9,6.1,6.3,6.1,6.4,6.6,6.8,6.7,6,5.7,5.5,5.5,5.8,6,5.4,6,6.7,6.3,5.6,5.5,5.5,6.1,5.8,5,5.6,5.7,5.7,6.2,5.1,5.7,6.3,5.8,7.1,6.3,6.5,7.6,4.9,7.3,6.7,7.2,6.5,6.4,6.8,5.7,5.8,6.4,6.5,7.7,7.7,6,6.9,5.6,7.7,6.3,6.7,7.2,6.2,6.1,6.4,7.2,7.4,7.9,6.4,6.3,6.1,7.7,6.3,6.4,6,6.9,6.7,6.9,5.8,6.8,6.7,6.7,6.3,6.5,6.2,5.9],"SepalWidthCm":[3.5,3,3.2,3.1,3.6,3.9,3.4,3.4,2.9,3.1,3.7,3.4,3,3,4,4.4,3.9,3.5,3.8,3.8,3.4,3.7,3.6,3.3,3.4,3,3.4,3.5,3.4,3.2,3.1,3.4,4.1,4.2,3.1,3.2,3.5,3.1,3,3.4,3.5,2.3,3.2,3.5,3.8,3,3.8,3.2,3.7,3.3,3.2,3.2,3.1,2.3,2.8,2.8,3.3,2.4,2.9,2.7,2,3,2.2,2.9,2.9,3.1,3,2.7,2.2,2.5,3.2,2.8,2.5,2.8,2.9,3,2.8,3,2.9,2.6,2.4,2.4,2.7,2.7,3,3.4,3.1,2.3,3,2.5,2.6,3,2.6,2.3,2.7,3,2.9,2.9,2.5,2.8,3.3,2.7,3,2.9,3,3,2.5,2.9,2.5,3.6,3.2,2.7,3,2.5,2.8,3.2,3,3.8,2.6,2.2,3.2,2.8,2.8,2.7,3.3,3.2,2.8,3,2.8,3,2.8,3.8,2.8,2.8,2.6,3,3.4,3.1,3,3.1,3.1,3.1,2.7,3.2,3.3,3,2.5,3,3.4,3],"PetalLengthCm":[1.4,1.4,1.3,1.5,1.4,1.7,1.4,1.5,1.4,1.5,1.5,1.6,1.4,1.1,1.2,1.5,1.3,1.4,1.7,1.5,1.7,1.5,1,1.7,1.9,1.6,1.6,1.5,1.4,1.6,1.6,1.5,1.5,1.4,1.5,1.2,1.3,1.5,1.3,1.5,1.3,1.3,1.3,1.6,1.9,1.4,1.6,1.4,1.5,1.4,4.7,4.5,4.9,4,4.6,4.5,4.7,3.3,4.6,3.9,3.5,4.2,4,4.7,3.6,4.4,4.5,4.1,4.5,3.9,4.8,4,4.9,4.7,4.3,4.4,4.8,5,4.5,3.5,3.8,3.7,3.9,5.1,4.5,4.5,4.7,4.4,4.1,4,4.4,4.6,4,3.3,4.2,4.2,4.2,4.3,3,4.1,6,5.1,5.9,5.6,5.8,6.6,4.5,6.3,5.8,6.1,5.1,5.3,5.5,5,5.1,5.3,5.5,6.7,6.9,5,5.7,4.9,6.7,4.9,5.7,6,4.8,4.9,5.6,5.8,6.1,6.4,5.6,5.1,5.6,6.1,5.6,5.5,4.8,5.4,5.6,5.1,5.1,5.9,5.7,5.2,5,5.2,5.4,5.1],"PetalWidthCm":[0.2,0.2,0.2,0.2,0.2,0.4,0.3,0.2,0.2,0.1,0.2,0.2,0.1,0.1,0.2,0.4,0.4,0.3,0.3,0.3,0.2,0.4,0.2,0.5,0.2,0.2,0.4,0.2,0.2,0.2,0.2,0.4,0.1,0.2,0.1,0.2,0.2,0.1,0.2,0.2,0.3,0.3,0.2,0.6,0.4,0.3,0.2,0.2,0.2,0.2,1.4,1.5,1.5,1.3,1.5,1.3,1.6,1,1.3,1.4,1,1.5,1,1.4,1.3,1.4,1.5,1,1.5,1.1,1.8,1.3,1.5,1.2,1.3,1.4,1.4,1.7,1.5,1,1.1,1,1.2,1.6,1.5,1.6,1.5,1.3,1.3,1.3,1.2,1.4,1.2,1,1.3,1.2,1.3,1.3,1.1,1.3,2.5,1.9,2.1,1.8,2.2,2.1,1.7,1.8,1.8,2.5,2,1.9,2.1,2,2.4,2.3,1.8,2.2,2.3,1.5,2.3,2,2,1.8,2.1,1.8,1.8,1.8,2.1,1.6,1.9,2,2.2,1.5,1.4,2.3,2.4,1.8,1.8,2.1,2.4,2.3,1.9,2.3,2.5,2.3,1.9,2,2.3,1.8],"Species":["Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica"]}
df = pd.DataFrame(a)
print(a)
