import csv
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import math


def str2value(str):
  if (str == 'Setosa'):
    str = '0'
  if (str == 'Versicolor'):
    str = '1'
  if (str == 'Virginica'):
    str = '2'

  return str



layer0 = 4
layer1 = 3
output1 = np.zeros(layer1, dtype=np.float)
weight01 = np.random.uniform(0, 1, size=(layer1, layer0))
updatew01 = np.zeros((layer1, layer0), dtype=np.float)


dataset = []


######### reading ##############################################################################################
with open('/Users/zhoufei/Desktop/FAT/Deep learning/iris.csv') as csvfile:
  rows = csv.reader(csvfile)
  for row in rows:
    if (rows.line_num == 1):
      continue
    else:
      dataset += row[0:4]
      dataset += str2value(row[4])

# print dataset

dataset = map(eval, dataset)
dataset = np.array(dataset)
dataset = np.reshape(dataset, (len(dataset)/5, 5))
# print (dataset.shape[1])

for i in range(layer0):
  dataset[:,i] = dataset[:,i] / sum(dataset[:,i])

# print dataset
# print sum(dataset)
# print np.reshape(dataset, (len(dataset)/4, 4))
######### Training ##############################################################################################
epoch = 0
lr = 0.9
fold = 0
x = 0
y = 0
training_num = []
testing_num = []
dataset_num = []

random.seed()

dataset_num = random.sample(range(0, dataset.shape[0]), dataset.shape[0])

while fold < 5:

  weight01 = np.random.uniform(0, 1, size=(layer1, layer0))

  for j in range(layer1):
    weight01[j] = weight01[j] * 1 / weight01[j].sum()

  x = fold * dataset.shape[0] * 1 / 5
  y += dataset.shape[0] * 1 / 5

  testing_num = dataset_num[x : y]

  training_num = list(set(dataset_num).difference(set(testing_num)))
  testing_num.sort()
  training_num.sort()

  while epoch <= 100000:
    number = random.choice(training_num)
    for j in range(layer1):
      output1[j] = 0
      for i in range(layer0):
        output1[j] += weight01[j][i] * dataset[number][i]

    pos = np.argmax(output1)  # int

    sum = 0
    for i in range(4):
      sum += dataset[number][i]

    for i in range(layer0):
      updatew01[pos][i] = 0
      updatew01[pos][i] = lr * (dataset[number][i]/(sum) - weight01[pos][i])
      weight01[pos][i] += updatew01[pos][i]



    epoch += 1
  ######### validation ##############################################################################################
  for num in testing_num:
    # print num
    for j in range(layer1):
      output1[j] = 0
      for i in range(layer0):
        output1[j] += weight01[j][i] * dataset[num][i]

    # print output1
    pos = np.argmax(output1)
    print fold+1, 'fold testing','dataset =', dataset[num][4],'testing =', pos

  print '\n'
  fold += 1
point = 0
