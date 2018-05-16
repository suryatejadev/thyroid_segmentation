from __future__ import print_function
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Concatenate, Conv2D, Conv3D, MaxPooling2D, Conv2DTranspose, Dropout, \
        BatchNormalization, merge, UpSampling2D, Cropping2D, ZeroPadding2D, Reshape, core, Convolution2D, Activation, Lambda
from keras import backend as K

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

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

def unet(height=None, width=None, channels=1, features=32, 
        depth=4, padding='same', temperature=1.0, 
        batchnorm=False, dropout=0.0):

    # Define the input
    inputs = Input((height, width, channels))
    dilation = (1,1)
    pool = True

    # Contracting path
    conv1, pool1 = conv_block(inputs, features   , padding, dropout, batchnorm, dilation, pool)
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

    return Model(inputs=[inputs], outputs=[conv10])


