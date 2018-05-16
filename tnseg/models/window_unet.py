from __future__ import print_function
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Concatenate, Conv2D, Conv3D, MaxPooling2D, Conv2DTranspose, Dropout, \
        BatchNormalization, merge, UpSampling2D, Cropping2D, ZeroPadding2D, Reshape, core, Convolution2D, Activation, Lambda
from keras import backend as K

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def conv_block(x, n_filt=64, padding='same', dropout=0.0, batchnorm=False, dilation=(1,1), pool=True):
    def conv_l(inp):

        conv = Conv2D(n_filt, (3, 3), padding=padding, dilation_rate=dilation)(inp)
        conv = Activation('relu')(conv)
        conv = BatchNormalization()(conv) if batchnorm else conv
        conv = Dropout(dropout)(conv) if dropout>0.0 else conv
        return conv

    conv = conv_l(x)
    conv = conv_l(conv)
    pool = MaxPooling2D(pool_size=(2, 2))(conv) if pool else conv
    return conv,pool

def upconv_block(x, x_conv, n_filt, padding='same', dropout=0.0, batchnorm=False):
    #up_conv = UpSampling2D(size=(2, 2), data_format="channels_last")(x)
    up_conv = Conv2DTranspose(n_filt, (2, 2), strides=(2, 2), padding=padding)(x)
    # crop x_conv
    if padding=='valid':
        ch, cw = get_crop_shape(x_conv, up_conv)
        x_conv = Cropping2D(cropping=(ch,cw), data_format="channels_last")(x_conv)
    up   = concatenate([up_conv, x_conv], axis=3)

    conv = Conv2D(n_filt, (3, 3), padding=padding, dilation_rate=(1,1))(up)
    conv = Activation('relu')(conv)
    conv = BatchNormalization()(conv) if batchnorm else conv
    conv = Dropout(dropout)(conv) if dropout>0.0 else conv

    conv = Conv2D(n_filt, (3, 3), padding=padding, dilation_rate=(1,1))(conv)
    conv = Activation('relu')(conv)
    conv = BatchNormalization()(conv) if batchnorm else conv
    conv = Dropout(dropout)(conv) if dropout>0.0 else conv
    return conv

def conv_block_window(x, n_filt=64, padding='same', dropout=0.0, batchnorm=False, dilation=(1,1), pool=True, window_size=3):
    def conv_l(inp):
        conv = Conv2D(n_filt, (3, 3), padding=padding, dilation_rate=dilation)(inp)
        conv = Activation('relu')(conv)
        conv = BatchNormalization()(conv) if batchnorm else conv
        conv = Dropout(dropout)(conv) if dropout>0.0 else conv
        return conv

    def conv_l_window(inp,n_window):
        conv = Conv3D(n_filt, (3, 3, n_window), padding=padding, dilation_rate=1)(inp)
        conv = Activation('relu')(conv)
        conv = BatchNormalization()(conv) if batchnorm else conv
        conv = Dropout(dropout)(conv) if dropout>0.0 else conv
        return conv

    conv = conv_l_window(x,window_size)
    #  conv = Lambda(lambda y: K.squeeze(y, axis=3))(conv)
    #  conv = conv_l(conv)
    #  pool = MaxPooling2D(pool_size=(2, 2))(conv) if pool else conv
    return conv,0#pool

# Window UNet implementation using 3d kernel on 1st layer
def unet_window_3d(img_rows, img_cols, init_filt=32, padding='same', dropout=0.0, batchnorm=False, dilation=(1,1), pool=True, window_size=3):

    # Define the input
    inputs = Input((320,448, window_size, 1))

    # Contracting path
    conv1, pool1 = conv_block_window(inputs, init_filt , padding, dropout, batchnorm, dilation, pool, window_size)
    #  conv2, pool2 = conv_block(pool1 , init_filt*2 , padding, dropout, batchnorm, dilation, pool)
    #  conv3, pool3 = conv_block(pool2 , init_filt*4 , padding, dropout, batchnorm, dilation, pool)
    #  conv4, pool4 = conv_block(pool3 , init_filt*8 , padding, dropout, batchnorm, dilation, pool)
    #  conv5, _     = conv_block(pool4 , init_filt*16, padding, dropout, batchnorm, dilation, pool)
    #
    #  # Expanding path
    #  conv6 = upconv_block(conv5, conv4, init_filt*8, padding, dropout, batchnorm)
    #  conv7 = upconv_block(conv6, conv3, init_filt*4, padding, dropout, batchnorm)
    #  conv8 = upconv_block(conv7, conv2, init_filt*2, padding, dropout, batchnorm)
    #  conv9 = upconv_block(conv8, conv1, init_filt*1, padding, dropout, batchnorm)
    #  conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv1])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    #model.compile(optimizer=Adam(lr=1e-5), loss=sparse_categorical_crossentropy, metrics=[dice_coef])

    return model

# Window UNet implementation 2D (use 2D kernel and sum conv outputs of frames to get feature maps)
def window_unet(height=None, width=None, features=32, padding='same', dropout=0.0, batchnorm=False, dilation=(1,1), pool=True, window_size=3, dilation=dilation):

    # Define the input
    dilation = tuple(map(int, dilation))
    inputs = Input((None, None, window_size))

    # Contracting path
    conv1, pool1 = conv_block(inputs, features , padding, dropout, batchnorm, dilation, pool)
    conv2, pool2 = conv_block(pool1 , features*2 , padding, dropout, batchnorm, dilation, pool)
    conv3, pool3 = conv_block(pool2 , features*4 , padding, dropout, batchnorm, dilation, pool)
    conv4, pool4 = conv_block(pool3 , features*8 , padding, dropout, batchnorm, dilation, pool)
    conv5, _     = conv_block(pool4 , features*16, padding, dropout, batchnorm, dilation, pool)

    # Expanding path
    conv6 = upconv_block(conv5, conv4, features*8, padding, dropout, batchnorm)
    conv7 = upconv_block(conv6, conv3, features*4, padding, dropout, batchnorm)
    conv8 = upconv_block(conv7, conv2, features*2, padding, dropout, batchnorm)
    conv9 = upconv_block(conv8, conv1, features*1, padding, dropout, batchnorm)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model


