import numpy as np
import os
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import math


recogonize = 0
def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

################ declaration ###########################################################################################
layer0 = 28 * 28
layer1 = 30
layer2 = 4

# weight01 = np.zeros((layer1, layer0), dtype=np.float)
# weight12 = np.zeros((layer2, layer1), dtype=np.float)
output1 = np.zeros(layer1, dtype=np.float)
output2 = np.zeros(layer2, dtype=np.float)
delta2 = np.zeros(layer2, dtype=np.float)
delta1 = np.zeros(layer1, dtype=np.float)
updatew12 = np.zeros((layer2, layer1), dtype=np.float)
updatew01 = np.zeros((layer1, layer0), dtype=np.float)
weight01 = np.random.normal(0, 1, size=(layer1, layer0))
weight12 = np.random.normal(0, 1, size=(layer2, layer1))
Ans = np.zeros(layer2, dtype=np.float)
training_dir = '/Users/zhoufei/desktop/training'
training_dir_list = (os.listdir(training_dir))
data_dir = np.zeros(layer2, dtype=basestring)
data_dir_list = np.zeros(layer2, dtype=basestring)
training_test = np.zeros(layer2, dtype=int)

time = 0
MSE = 0
learningrate = 0.5
f = open('/Users/zhoufei/desktop/Answer.txt', 'w')
time = 0
data_sum = 0
print (test)
################ data reading ##########################################################################################
training_list.sort()
print (training_list)
print len(training_list)
# np.random.shuffle(training_list)
# print (training_list)
print ('\n')

for num in range(len(training_list)):

    training_dir[num] = os.path.join(training, training_list[num])
    training_data_list[num] = os.listdir(training_dir[num])
    training_data_list[num].sort()
    # np.random.shuffle(training_data_list[num])
    # print training_dir[i]
    print(training_list[num])
    print training_data_list[num]
    print (len(training_data_list[num]))

    # trainingdata = (len(training_data_list[0]) * 8 / 10)
    # print (trainingdata)

################ reading weight #############################################################################################
fW = open('/Users/zhoufei/desktop/W.txt', 'r')



for j in range(layer2):
    for i in range(layer1):
        weight12[j][i] = float(fW.readline())

for j in range(layer1):
    for x in range(28):
        for y in range(28):
            weight01[j][28 * x + y] = float(fW.readline())


################# testing declaration ###############################################################################################
test = np.zeros(layer2, dtype=int)
test_dir = '//Volumes/ADATA HD330/testing'
test_dir_list = (os.listdir(test_dir))
print len(test_dir_list)

f = open('/Users/zhoufei/desktop/Answer.txt', 'w')
time = 0
data_sum = 0
################ data reading #############################################################################################

for pic in range(len(test_dir_list)):
    # print (pic)
    img = mpimg.imread(os.path.join(test_dir, test_dir_list[pic]))
    # print training_dir_list[0]
    # print data_dir_list[0][pic]
    # print (img)

    for i in range(28):
        for j in range(28):
            if img[i][j] == 0:
                img[i][j] -= 0.5



    ################ Forward 01 ##################################################################################################
    count = 0

    for j in range(layer1):
        output1[j] = 0
        for x in range(28):
            for y in range(28):
                output1[j] += weight01[j][28 * x + y] * img[x][y]
                count += 1
                # print(output1[j], count)
        output1[j] = sigmoid(output1[j])
        # count += 1
        # print(output1[j], count)

    ################ Forward 12 #############################################################################################
    count = 0

    for j in range(layer2):
        output2[j] = 0
        for i in range(layer1):
            output2[j] += weight12[j][i] * output1[i]
            # count += 1
            # print(output2[j], count)

        output2[j] = sigmoid(output2[j])
        # print(output2[j], count)
        # count += 1


    recogonize = output2.tolist()
    # print (recogonize)
    recogonize = recogonize.index(max(recogonize))
    # print (recogonize)

    # if recogonize == num:
    #     test[num] += 1

    f.write(test_dir_list[pic][:-4] + ' ' + str(training_dir_list[recogonize]) + '\n')


    data_sum += 1
#
# print (test)
# print (sum(test))
# print (data_sum)
# print (sum(test) / float(data_sum))


f.close()
fW.close()
