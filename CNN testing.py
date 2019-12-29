from keras.models import load_model
import numpy as np
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, MaxPooling2D, Conv2D, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.preprocessing import image
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.image as mpimg
import os
from keras.preprocessing.image import ImageDataGenerator
# del model

model = load_model('model.h5')

test_dir = '/Volumes/ADATA HD330/hw2/02468'
test_dir_list = (os.listdir(test_dir))
fW = open('/Users/zhoufei/desktop/answer.txt', 'w')

test_dir_list.sort()

for num in range(len(test_dir_list)):
    img_path = os.path.join(test_dir, test_dir_list[num])
    img = image.load_img(img_path, target_size=(28, 28))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255.
    predictions = model.predict_classes(img) * 2
    # print(predictions)
    fW.write(test_dir_list[num][:-4] + ' ' + str(predictions[0]) + '\n')
