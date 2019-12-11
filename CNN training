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
training_dir = '/Users/zhoufei/desktop/training'
validation_dir = '/Users/zhoufei/desktop/validation'
# fuck_dir = '/Users/zhoufei/desktop/simulation/fuck'
# fuck_dir_list = (os.listdir(fuck_dir))
# fW = open('/Users/zhoufei/desktop/answer.txt', 'w')





########### model ##############
model = Sequential()
########### input layer, hidden layer, output layer ##############
model.add(Conv2D(32, 3, 3, input_shape = (28, 28, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 5, activation = 'softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])






# All images will be rescaled by 1./255
training_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
training_generator = training_datagen.flow_from_directory(
        training_dir,  # This is the source directory for training images
        target_size=(28, 28),  # All images will be resized to 150x150
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(28, 28),
        batch_size=20,
        class_mode='categorical')

history = model.fit_generator(
      training_generator,
      steps_per_epoch=100,  # 3200 images = batch_size * steps
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,  # 800 images = batch_size * steps
      verbose=2)


from keras.models import load_model
model.save('model.h5')  # creates a HDF5 file 'model.h5'

# for num in range(len(fuck_dir_list)):
#     img_path = os.path.join(fuck_dir, fuck_dir_list[num])
#     img = image.load_img(img_path, target_size=(28, 28))
#     img = image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = img/255.
#     predictions = model.predict_classes(img)
#     print(predictions)
#     fW.write(str(predictions[0]) + '\n')



