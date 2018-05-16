import numpy as np
import matplotlib.pyplot as plt
import os
import pydicom as pyd
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import bisect
import random
import math
from Augmentor import *

def import_dicom_data(path):
    data_path = path + '/images/'
    annot_path = path + '/labels/'
    data_list = glob(data_path + '*.dcm')
    annot_list = glob(annot_path + '*.dcm')
    N = len(data_list)
    data = []
    annot = []
    annot_frames = np.zeros((N))
    print('Data Image Resolutions')
    for i in range(N):
        x = pyd.read_file(data_list[i]).pixel_array
        x = x[:len(x) / 2]
        y = pyd.read_file(annot_list[i]).pixel_array
        y = y[:len(y) / 2]
        n_frame = 0
        for j in range(y.shape[0]):
            if np.where(y[j] == 1)[0].size > 0:
                n_frame += 1
        annot_frames[i] = n_frame
        print(x.shape, n_frame)
        data.append(x)
        annot.append(y)
    return data, annot

def zeropad(data, annot, h_max, w_max):
    # If the data is a list of images of different resolutions
    # useful in testing
    if isinstance(data, list):
        n = len(data)
        data_pad = np.zeros((n, h_max, w_max))
        annot_pad = np.zeros((n, h_max, w_max))
        for i in range(n):
            pad_l1 = (h_max - data[i].shape[0]) // 2
            pad_l2 = (h_max - data[i].shape[0]) - (h_max - data[i].shape[0]) // 2
            pad_h1 = (w_max - data[i].shape[1]) // 2
            pad_h2 = (w_max - data[i].shape[1]) - (w_max - data[i].shape[1]) // 2
            data_pad[i] = np.pad(data[i], ((pad_l1, pad_l2), (pad_h1, pad_h2)), 'constant',
                                 constant_values=((0, 0), (0, 0)))
            annot_pad[i] = np.pad(annot[i], ((pad_l1, pad_l2), (pad_h1, pad_h2)), 'constant',
                                  constant_values=((0, 0), (0, 0)))
    # If data is a numpy array with images of same resolution
    else:
        pad_l1 = (h_max - data.shape[1]) // 2
        pad_l2 = (h_max - data.shape[1]) - (h_max - data.shape[1]) // 2
        pad_h1 = (w_max - data.shape[2]) // 2
        pad_h2 = (w_max - data.shape[2]) - (w_max - data.shape[2]) // 2

        data_pad = np.pad(data, ((0, 0), (pad_l1, pad_l2), (pad_h1, pad_h2)), 'constant',
                          constant_values=((0, 0), (0, 0), (0, 0)))
        annot_pad = np.pad(annot, ((0, 0), (pad_l1, pad_l2), (pad_h1, pad_h2)), 'constant',
                           constant_values=((0, 0), (0, 0), (0, 0)))
    return data_pad, annot_pad

def data_augment(imgs, lb):
    p = Pipeline()
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    imgs_temp, lb_temp = np.zeros(imgs.shape), np.zeros(imgs.shape)
    for i in range(imgs.shape[0]):
        pil_images = p.sample_with_array(imgs[i], ground_truth=lb[i], mode='L')
        imgs_temp[i], lb_temp[i] =  np.asarray(pil_images[0]), np.asarray(pil_images[1])

    return imgs_temp, lb_temp

def get_weighted_batch(imgs, labels, batch_size, data_aug, high_skew=False):
    while 1:
        thy_re = [np.count_nonzero(labels[i] == 1) * 1.0 / np.prod(labels[i].shape) for i in range(imgs.shape[0])]
        if high_skew==True:
            thy_re = [el**2 for el in thy_re]
        cumul = [thy_re[0]]
        for item in thy_re[1:]: cumul.append(cumul[-1] + item)
        total_prob = sum(thy_re)

        ar_inds = [bisect.bisect_right(cumul, random.uniform(0, total_prob)) for i in range(batch_size)]
        lb, batch_imgs = labels[ar_inds], imgs[ar_inds]
        l, r, t, b = 0, batch_imgs.shape[1], 0, batch_imgs.shape[2]
        for i in range(batch_imgs.shape[1]):
            if np.all(batch_imgs[:, i, :] == 0):
                l = i + 1
            else:
                break
        for i in range(batch_imgs.shape[1] - 1, -1, -1):
            if np.all(batch_imgs[:, i, :] == 0):
                r = i
            else:
                break
        for i in range(batch_imgs.shape[2]):
            if np.all(batch_imgs[:, :, i] == 0):
                t = i + 1
            else:
                break
        for i in range(batch_imgs.shape[2] - 1, -1, -1):
            if np.all(batch_imgs[:, :, i] == 0):
                b = i
            else:
                break
        l, r, t, b = (l // 16) * 16, math.ceil(r * 1.0 / 16) * 16, (t // 16) * 16, math.ceil(b * 1.0 / 16) * 16
        l, r, t, b = int(l), int(r), int(t), int(b)
        batch_imgs, lb = batch_imgs[:, l:r, t:b], lb[:, l:r, t:b]
        if (data_aug):
            batch_imgs, lb =  data_augment(batch_imgs, lb)
        yield np.expand_dims(batch_imgs, axis=3),np.expand_dims(lb, axis=3)

def get_weighted_batch_window_2d(imgs, labels, batch_size, data_aug, n_window=0, high_skew=False):
    # a=0
    # if a==0:
    # print('datagen')
    while 1:
        thy_re = [np.count_nonzero(labels[i] == 1) * 1.0 / np.prod(labels[i].shape) for i in range(imgs.shape[0])]
        if high_skew==True:
            thy_re = [el**2 for el in thy_re]
        cumul = [thy_re[0]]
        for item in thy_re[1:]: cumul.append(cumul[-1] + item)
        total_prob = sum(thy_re)

        ar_inds = [bisect.bisect_right(cumul, random.uniform(0, total_prob)) for i in range(batch_size)]
        if n_window==0:
            batch_imgs = imgs[ar_inds]
        # Get n_window frames per index.
        else:
            batch_imgs = np.zeros((batch_size*n_window,imgs.shape[1],imgs.shape[2]))
            for i in range(batch_size):
                if ar_inds[i]==0:
                    ar_inds[i] = 1
                elif ar_inds[i] == len(imgs)-1:
                    ar_inds[i] -= 1
                batch_imgs[n_window*i:n_window*(i+1)] = imgs[ar_inds[i]-1:ar_inds[i]+2]
        lb = labels[ar_inds]
        l, r, t, b = 0, batch_imgs.shape[1], 0, batch_imgs.shape[2]
        for i in range(batch_imgs.shape[1]):
            if np.all(batch_imgs[:, i, :] == 0):
                l = i + 1
            else:
                break
        for i in range(batch_imgs.shape[1] - 1, -1, -1):
            if np.all(batch_imgs[:, i, :] == 0):
                r = i
            else:
                break
        for i in range(batch_imgs.shape[2]):
            if np.all(batch_imgs[:, :, i] == 0):
                t = i + 1
            else:
                break
        for i in range(batch_imgs.shape[2] - 1, -1, -1):
            if np.all(batch_imgs[:, :, i] == 0):
                b = i
            else:
                break
        l, r, t, b = (l // 16) * 16, math.ceil(r * 1.0 / 16) * 16, (t // 16) * 16, math.ceil(b * 1.0 / 16) * 16
        l, r, t, b = int(l), int(r), int(t), int(b)
        batch_imgs, lb = batch_imgs[:, l:r, t:b], lb[:, l:r, t:b]
        # batch_imgs_3d = np.zeros((batch_size,imgs.shape[1], imgs.shape[2], n_window))
        # k=0
        # for i in range(batch_size):
            # for j in range(n_window):
                # batch_imgs_3d[i,:,:,j] = batch_imgs[k,:,:]
                # k += 1
        batch_imgs = np.array([np.rollaxis(batch_imgs[n_window*i:n_window*(i+1)],0,3) for i in range(batch_size)])
        if (data_aug):
            batch_imgs, lb =  data_augment(batch_imgs, lb)
        # print('batch = ',batch_imgs.shape, lb.shape)
        yield batch_imgs,np.expand_dims(lb, axis=3)

def get_max_dimensions(data_list):
    return 320, 448

def create_generators(datadir=None, batch_size=64, augmentation_args=None,\
        model='unet', zero_padding=[0,0], data_skew=False, validation_index=None):
    
    # Load data from the data directory
    if datadir==None:
        raise Exception("Data directory not specified")
    data_list, annot_list = import_dicom_data(datadir)
    print(len(data_list))
    
    # Get the max dimensions of the DICOM frames, and zeropad all images
    h_max, w_max = get_max_dimensions(data_list)
    for i, data in enumerate(data_list):
        data_list[i], annot_list[i] = zeropad(data_list[i], annot_list[i], h_max, w_max)

    # Get train and validation data
    N = len(data_list)
    if validation_index==None:
        raise Exception("Please specify validation indices")
    else:
        trn_imgs = []
        trn_labels = []
        val_imgs = []
        val_labels = []
        for i in range(len(data_list)):
            if i in validation_index:
                val_imgs.append(data_list[i])
                val_labels.append(annot_list[i])
            else:
                trn_imgs.append(data_list[i])
                trn_labels.append(annot_list[i])
        val_imgs = np.concatenate(val_imgs,axis=0)
        val_labels = np.concatenate(val_labels,axis=0)
        trn_imgs = np.concatenate(trn_imgs,axis=0)
        trn_labels = np.concatenate(trn_labels,axis=0)
        print(val_imgs.shape, val_labels.shape, trn_imgs.shape, trn_labels.shape)

    # Data generator for augmentation
    if augmentation_args !=None:
        data_augment=True
        datagen = ImageDataGenerator(
            rotation_range=augmentation_args['rotation_range'],
            width_shift_range=augmentation_args['width_shift_range'],
            height_shift_range=augmentation_args['height_shift_range'],
            shear_range=augmentation_args['shear_range'],
            zoom_range=augmentation_args['zoom_range'],
            fill_mode=augmentation_args['fill_mode'])
    else:
        data_augment=False
        datagen = ImageDataGenerator(
            rotation_range=0.,
            width_shift_range=0.,
            height_shift_range=0.,
            shear_range=0.,
            zoom_range=0.,
            horizontal_flip=0.,
            fill_mode=0.)
    
    # Get model specific data generators
    if model in ['unet', 'dilated-unet', 'dilated-densenet']:
        if data_skew==True:
            train_generator = get_weighted_batch(trn_imgs, trn_labels, batch_size, data_augment)
            val_generator = get_weighted_batch(val_imgs, val_labels, batch_size, data_augment, high_skew=True)
        else:
            train_generator = datagen.flow(x=np.expand_dims(trn_imgs, axis=3), y=np.expand_dims(trn_labels, axis=3), batch_size=16)
            val_generator = datagen.flow(x=np.expand_dims(val_imgs, axis=3), y=np.expand_dims(val_labels, axis=3), batch_size=16)
    elif mode=='window-unet':
        train_generator = get_weighted_batch_window_2d(trn_imgs, trn_labels, batch_size, data_augment, window)
        val_generator = get_weighted_batch_window_2d(val_imgs, val_labels, batch_size, data_augment, window, high_skew=True)

    return train_generator, val_generator
