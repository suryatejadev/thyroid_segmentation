#!/usr/bin/env python

from __future__ import division, print_function

import os
import argparse
import logging
import sys
sys.path.append('..')
import matplotlib as mpl
mpl.use('Agg')

from keras import losses, optimizers, utils
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from tnseg import dataset, models, loss, opts, evaluate

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

def select_optimizer(optimizer_name, optimizer_args):
    optimizers = {
        'sgd': SGD,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'adam': Adam,
        'adamax': Adamax,
        'nadam': Nadam,
    }
    if optimizer_name not in optimizers:
        raise Exception("Unknown optimizer ({}).".format(name))

    return optimizers[optimizer_name](**optimizer_args)

def train(validation_index, args):


    logging.info("Loading dataset...")
    augmentation_args = None
    if args.data_augment==True:
        augmentation_args = {
            'rotation_range': args.rotation_range,
            'width_shift_range': args.width_shift_range,
            'height_shift_range': args.height_shift_range,
            'shear_range': args.shear_range,
            'zoom_range': args.zoom_range,
            'fill_mode' : args.fill_mode,
            'alpha': args.alpha,
            'sigma': args.sigma,
        }

    #  train_generator, train_steps_per_epoch, \
    #      val_generator, val_steps_per_epoch = dataset.create_generators(
    #          args.datadir, args.batch_size,
    #          validation_split=args.validation_split,
    #          mask=args.classes,
    #          shuffle_train_val=args.shuffle_train_val,
    #          shuffle=args.shuffle,
    #          seed=args.seed,
    #          normalize_images=args.normalize,
    #          augment_training=args.augment_training,
    #          augment_validation=args.augment_validation,
    #          augmentation_args=augmentation_args, # new arguments from here...
    #          window_size=0,
    #          adaptive_padding=True,
    #          constant_padding_height=320,
    #          constant_padding_width=448,
    #          datagen_method='zeropad'
    #          )

    train_generator, val_generator = dataset.create_generators(
            args.datadir, args.batch_size,
            augmentation_args=augmentation_args,
            model=args.model,
            zero_padding=args.zero_padding,
            data_skew=args.data_skew,
            validation_index=validation_index,
            window=args.window
            )

    if args.model=='unet':
        m = models.unet(height=None, width=None, channels=1, features=args.features,
                depth=args.depth, padding=args.padding, temperature=args.temperature,
                batchnorm=args.batchnorm, dropout=args.dropout)
    elif args.model=='dilated-unet':
        m = models.dilated_unet(height=None, width=None, channels=1,
                classes=2, features=args.features, depth=args.depth,
                temperature=args.temperature, padding=args.padding,
                batchnorm=args.batchnorm, dropout=args.dropout)
    elif args.model=='dilated-densenet':
        m = models.dilated_densenet(height=None, width=None, channels=1,
                classes=2, features=args.features, depth=args.depth,
                temperature=args.temperature, padding=args.padding,
                batchnorm=args.batchnorm,dropout=args.dropout)
    elif args.model=='window-unet':
        m = models.window_unet(height=None, width=None,
                features=args.features, padding=args.padding,
                dropout=args.dropout, batchnorm=args.batchnorm, window_size=args.window)
    else:
        raise ValueError('Model not supported. Please select from: unet,\
                dilated-unet, dilated-densenet, window-unet')

        m.summary()

    if args.load_weights:
        logging.info("Loading saved weights from file: {}".format(args.load_weights))
        m.load_weights(args.load_weights)

    # instantiate optimizer, and only keep args that have been set
    # (not all optimizers have args like `momentum' or `decay')
    optimizer_args = {
        'lr':       args.learning_rate,
        'momentum': args.momentum,
        'decay':    args.decay
    }
    for k in list(optimizer_args):
        if optimizer_args[k] is None:
            del optimizer_args[k]
    optimizer = select_optimizer(args.optimizer, optimizer_args)

    # select loss function: pixel-wise crossentropy, soft dice or soft
    # jaccard coefficient
    #  if args.loss == 'pixel':
    #      def lossfunc(y_true, y_pred):
    #          return loss.weighted_categorical_crossentropy(
    #              y_true, y_pred, args.loss_weights)
    #  elif args.loss == 'dice':
    #      def lossfunc(y_true, y_pred):
    #          return loss.sorensen_dice_loss(y_true, y_pred, args.loss_weights)
    #  elif args.loss == 'jaccard':
    #      def lossfunc(y_true, y_pred):
    #          return loss.jaccard_loss(y_true, y_pred, args.loss_weights)
    #  else:
    #      raise Exception("Unknown loss ({})".format(args.loss))

    #  def dice(y_true, y_pred):
    #      batch_dice_coefs = loss.sorensen_dice(y_true, y_pred, axis=[1, 2])
    #      dice_coefs = K.mean(batch_dice_coefs, axis=0)
    #      return dice_coefs[1]    # HACK for 2-class cas   metrics = [loss.dice_coef]

    if args.loss == 'dice':
        lossfunc = lambda y_true, y_pred: loss.dice_coef_loss(y_true, y_pred)
    elif args.loss == 'pixel':
        lossfunc = lambda y_true, y_pred: loss.bin_crossentropy_loss(y_true, y_pred)
    else:
        raise Exception("Unknown loss ({})".format(args.loss))

    metrics = [loss.dice_coef]

    m.compile(optimizer=optimizer, loss=lossfunc, metrics=metrics)

    # automatic saving of model during training
    #  if args.checkpoint:
    #      if args.loss == 'pixel':
    #          filepath = os.path.join(
    #              args.outdir, "weights-{epoch:02d}-{val_acc:.4f}.hdf5")
    #          monitor = 'val_acc'
    #          mode = 'max'
    #      elif args.loss == 'dice':
    #          filepath = os.path.join(
    #              args.outdir, "weights-{epoch:02d}-{val_dice:.4f}.hdf5")
    #          monitor='val_dice'
    #          mode = 'max'
    #      elif args.loss == 'jaccard':
    #          filepath = os.path.join(
    #              args.outdir, "weights-{epoch:02d}-{val_jaccard:.4f}.hdf5")
    #          monitor='val_jaccard'
    #          mode = 'max'
    #      checkpoint = ModelCheckpoint(
    #          filepath, monitor=monitor, verbose=1,
    #          save_best_only=True, mode=mode)
    #      callbacks = [checkpoint]
    #  else:
    #      callbacks = []

    # train
    if args.checkpoint:
        wt_index = str(validation_index[0]/2)
        wt_path = args.outdir + '/weights/wt-'+wt_index+'-{epoch:02d}-{val_dice_coef:.2f}.h5'
        checkpoint = ModelCheckpoint(wt_path, monitor='val_dice_coef', verbose=1,
                                     save_weights_only=True, period=args.ckpt_period)
        callbacks = [checkpoint]
    else:
        callbacks = []
    logging.info("Begin training.")
    out = m.fit_generator(train_generator,
                    epochs=args.epochs,
                    steps_per_epoch=args.train_steps_per_epoch,
                    validation_data=val_generator,
                    validation_steps=args.val_steps_per_epoch,
                    callbacks=callbacks,
                    verbose=2)
    return m, out

def evaluation(model, out, validation_index, args):
    iter_model = validation_index[0]/2

    # Paths to save predictions and results of the model ---------------------
    save_prediction_path = args.outdir+'/predictions/'
    save_results_path = args.outdir+'/results/'

    # Saving accuracy and error plots ----------------------------------------
    evaluate.eval_error_plots(out, save_results_path+str(iter_model)+'_')

    # Saving history ---------------------------------------------------------
    results_dict = {}
    results_dict['history_'+str(iter_model)] = out.history

    # Saving output annotation maps ------------------------------------------
    folder_names = os.listdir(args.datadir+'/images/')
    dice_vals = []
    for folder_index in validation_index:
        folder = folder_names[folder_index][:-4]
        folder_prediction_path = save_prediction_path+folder+'/'
        if os.path.exists(folder_prediction_path)==False:
            os.mkdir(folder_prediction_path)
        dice_vals.append(evaluate.evaluate_test_folder(model, folder_prediction_path,
            args.datadir+'/data_images/'+folder+'/', n_window=args.window))
    return dice_vals

####################################################################
# Training Methodology:
# - The dataset has 16 DICOM videos
# - Using each architecture, We are building 8 models,
#   using 2 videos for validation for each model
# - Our output is the validation dice coefficient for the 16 videos
# - This methodology is adopted due to the availability of less data
##########################################################3#########
if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    args = opts.parse_arguments()

    # Creating experiment output folders
    if os.path.exists(args.outdir)==False:
        os.mkdir(args.outdir)
    for item in ['predictions/', 'results/', 'weights/']:
        path = args.outdir + '/' + item
        if os.path.exists(path) == False:
            os.mkdir(path)

    # Train and evaluate
    dice_vals = {}
    for iter_model in range(1):
        validation_index = [iter_model*2, iter_model*2+1]
        print('Validation Folders = ',validation_index)
        model, out = train(validation_index, args)
        dice_vals['D'+str(iter_model*2+1)],\
                dice_vals['D'+str(iter_model*2+2)] = \
                evaluation(model, out, validation_index, args)

    # Print the Folder Dice coefficients in a file
    dice_coef_path = args.outdir+'/results/val_dice.txt'
    with open(args.outdir+'/results/val_dice.txt','w') as f:
        f.write(str(dice_vals))
    f.close()

