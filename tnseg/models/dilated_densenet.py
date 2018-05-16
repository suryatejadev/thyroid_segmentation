from __future__ import print_function
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Concatenate, Conv2D, Conv3D, MaxPooling2D, Conv2DTranspose, Dropout, \
        BatchNormalization, merge, UpSampling2D, Cropping2D, ZeroPadding2D, Reshape, core, Convolution2D, Activation, Lambda
from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def dilated_densenet1(height, width, channels, classes, features=12, depth=4,
                     temperature=1.0, padding='same', batchnorm=False,
                     dropout=0.0, dilation=dilation):
    dilation = tuple(map(int, dilation))
    x = Input(shape=(height, width, channels))
    inputs = x

    # initial convolution
    x = Conv2D(features, kernel_size=(5,5), padding=padding)(x)

    maps = [inputs]
    dilation_rate = 1
    kernel_size = (3,3)
    for n in range(depth):
        maps.append(x)
        x = Concatenate()(maps)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(features, kernel_size, dilation_rate=dilation_rate,
                   padding=padding)(x)
        dilation_rate *= 2

    probabilities = Conv2D(1, kernel_size=(1,1), activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=probabilities)
    return model

def dilated_densenet2(height, width, channels, classes, features=12, depth=4,
                      temperature=1.0, padding='same', batchnorm=False,
                      dropout=0.0, dilation=dilation):
    dilation = tuple(map(int, dilation))
    x = Input(shape=(height, width, channels))
    inputs = x

    # initial convolution
    x = Conv2D(features, kernel_size=(5,5), padding=padding)(x)

    maps = [inputs]
    dilation_rate = 1
    kernel_size = (3,3)
    for n in range(depth):
        maps.append(x)
        x = Concatenate()(maps)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(4*features, kernel_size=1)(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(features, kernel_size, dilation_rate=dilation_rate,
                   padding=padding)(x)
        dilation_rate *= 2

    probabilities = Conv2D(1, kernel_size=(1,1), activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=probabilities)
    return model

def dilated_densenet(height, width, channels, classes, features=12, depth=4,
                      temperature=1.0, padding='same', batchnorm=False,
                      dropout=0.0, dilation=dilation):
    dilation = tuple(map(int, dilation))
    x = Input(shape=(height, width, channels))
    inputs = x

    # initial convolution
    x = Conv2D(features, kernel_size=(5,5), padding=padding)(x)

    maps = [inputs]
    dilation_rate = 1
    kernel_size = (3,3)
    for n in range(depth):
        maps.append(x)
        x = Concatenate()(maps)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(features, kernel_size, dilation_rate=dilation_rate,
                   padding=padding)(x)
        dilation_rate *= 2

    # Additional 2 layers to help generate segmentation mask
    x = Conv2D(features, kernel_size=(3,3), activation='relu', padding=padding)(x)
    x = Conv2D(features, kernel_size=(3,3), activation='relu', padding=padding)(x)

    probabilities = Conv2D(1, kernel_size=(1,1), activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=probabilities)
    return model



