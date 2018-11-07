import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint


IMAGE_LIB = str(sys.argv[1])+'/input/2d_images/'
MASK_LIB = str(sys.argv[1])+'/input/2d_masks/'
IMG_HEIGHT, IMG_WIDTH = 32, 32
SEED=42

all_images = [x for x in sorted(os.listdir(IMAGE_LIB)) if x[-4:] == '.tif']

x_data = np.empty((len(all_images), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
for i, name in enumerate(all_images):
    im = cv2.imread(IMAGE_LIB + name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    x_data[i] = im

y_data = np.empty((len(all_images), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
for i, name in enumerate(all_images):
    im = cv2.imread(MASK_LIB + name, cv2.IMREAD_UNCHANGED).astype('float32')/255.
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    y_data[i] = im


x_data = x_data[:,:,:,np.newaxis]
y_data = y_data[:,:,:,np.newaxis]
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size = 0.5)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


input_layer = Input(shape=x_train.shape[1:])
c1 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
l = MaxPool2D(strides=(2, 2))(c1)
c2 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2, 2))(c2)
c3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2, 2))(c3)
c4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2, 2))(c4), c3], axis=-1)
l = Conv2D(filters=32, kernel_size=(2, 2), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2, 2))(l), c2], axis=-1)
l = Conv2D(filters=24, kernel_size=(2, 2), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2, 2))(l), c1], axis=-1)
l = Conv2D(filters=16, kernel_size=(2, 2), activation='relu', padding='same')(l)
l = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(l)
l = Dropout(0.5)(l)
output_layer = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(l)

model = Model(input_layer, output_layer)

def my_generator(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch

image_batch, mask_batch = next(my_generator(x_train, y_train, 8))
model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef])

weight_saver = ModelCheckpoint('lung.h5', monitor='val_dice_coef',
                                              save_best_only=True, save_weights_only=True)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)

hist = model.fit_generator(my_generator(x_train, y_train, 8),
                           steps_per_epoch = 200,
                           validation_data = (x_val, y_val),
                           epochs=10, verbose=2,
                           callbacks = [weight_saver, annealer])