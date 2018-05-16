from __future__ import division
from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.losses import binary_crossentropy

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_numpy(y_true, y_pred):
    smooth = 1.
    y_true_f = np.ndarray.flatten(y_true)
    y_pred_f = np.ndarray.flatten(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def bin_crossentropy_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)

def iou_score(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (1. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_coef_log_loss(y_true, y_pred):
    return -K.log(dice_coef(y_true, y_pred))

def iou_score_loss(y_true, y_pred):
    return -iou_score(y_true, y_pred)


