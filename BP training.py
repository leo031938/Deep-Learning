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
fW = open('/Users/zhoufei/desktop/W.txt', 'w')

################ data reading ##########################################################################################
training_dir_list.sort()
print (training_dir_list)
print len(training_dir_list)
# np.random.shuffle(training_dir_list)
# print (training_dir_list)
print ('\n')

for num in range(len(training_dir_list)):

    data_dir[num] = os.path.join(training_dir, training_dir_list[num])
    data_dir_list[num] = os.listdir(data_dir[num])
    data_dir_list[num].sort()
    # np.random.shuffle(data_dir_list[num])
    # print data_dir[i]
    print(training_dir_list[num])
    print data_dir_list[num]
    print (len(data_dir_list[num]))

    # trainingdata = (len(data_dir_list[0]) * 8 / 10)
    # print (trainingdata)


while time < 10000 :
    random.seed()
    num = random.randint(0, len(training_dir_list) - 1)
    pic = random.randint(0, len(data_dir_list[num]) - 1)
    # print (pic)
    img = mpimg.imread(os.path.join(training_dir, training_dir_list[num], data_dir_list[num][pic]))
    # print training_dir_list[0]
    # print data_dir_list[0][pic]
    # print (img)

    for i in range(28):
        for j in range(28):
            img[i][j] -= 0.5

    for j in range(layer2):
        if j == num:
            Ans[j] = 1

        else:
            Ans[j] = 0

    ################ training ##############################################################################################


    ################ Forward 01 #############################################################################################
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

    ################ Back path delta ##########################################################################################
    count = 0

    for j in range(layer2):
        delta2[j] = 0
        delta2[j] = (Ans[j] - output2[j]) * output2[j] * (1 - output2[j])
        # count += 1
        # print(delta2[j], count)

    for i in range(layer1):
        delta1[i] = 0
        for j in range(layer2):
            delta1[i] += weight12[j][i] * delta2[j] * output1[i] * (1 - output1[i])
            # count += 1
            # print(delta1[j], count)
    ################ Back path 21 ##########################################################################################

    for j in range(layer2):
        for i in range(layer1):
            updatew12[j][i] = learningrate * delta2[j] * output1[i]
            weight12[j][i] += updatew12[j][i]


    ################ Back path 10 ##########################################################################################

    for j in range(layer1):
        for x in range(28):
            for y in range(28):
                updatew01[j][28 * x + y] = learningrate * delta1[j] * img[x][y]
                weight01[j][28 * x + y] += updatew01[j][28 * x + y]



    ################ MSE ##########################################################################################
    MSE = 0

    for j in range(layer2):
        MSE += (pow(Ans[j] - output2[j], 2) / layer2)

    # print (MSE, output2[num])
    ################ validation ##########################################################################################
    recogonize = output2.tolist()
    # print (recogonize)
    recogonize = recogonize.index(max(recogonize))
    # print (recogonize)

    if recogonize == num:
        training_test[num] += 1

    time += 1

print (training_test)

for j in range(layer2):
    for i in range(layer1):
        fW.write(str(weight12[j][i]) + '\n')


for j in range(layer1):
    for x in range(28):
        for y in range(28):
            fW.write(str(weight01[j][28 * x + y]) + '\n' )

fW.close()
