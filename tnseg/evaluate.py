import numpy as np
import sys
sys.path.append('.')

from dataset import *
from scipy.misc import imsave
import matplotlib.pyplot as plt
import os
import pdb
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
from loss import *
from tqdm import tqdm

from ufarray import *
sys.setrecursionlimit(10000)

# save error plots
def eval_error_plots(out, output_dir):
    # get train and val acc and loss
    loss = out.history['loss']
    val_loss = out.history['val_loss']
    acc_key = [i for i in out.history.keys() if ('val' not in i and 'loss' not in i)][0]
    acc = out.history[acc_key]
    val_acc = out.history['val_' + acc_key]

    # Plot and save them
    plt.figure()
    plt.plot(loss, 'b', label='Training')
    plt.plot(val_loss, 'r', label='Validation')
    plt.title('Training vs Validation loss')
    plt.legend()
    plt.savefig(output_dir + 'plot_loss.png', dpi=300)
    plt.close()
    plt.figure()
    plt.plot(acc, 'b', label='Training')
    plt.plot(val_acc, 'r', label='Validation')
    plt.title('Training vs Validation ' + acc_key)
    plt.legend()
    plt.savefig(output_dir + 'plot_accuracy.png', dpi=300)
    plt.close()


def post_processing(data, probas):
    [n,h,w] = data.shape
    n_labels = 2
    pred_maps = np.zeros(data.shape)
    print 'postprocessing:', data.shape, probas.shape
    for i in tqdm(range(n)):
        img = data[i][...,np.newaxis]
        proba = probas[i]
        labels = np.zeros((2,img.shape[0],img.shape[1]))
        labels[0] = 1-proba
        labels[1] = proba

        U = unary_from_softmax(labels)  # note: num classes is first dim
        pairwise_energy = create_pairwise_bilateral(sdims=(50,50), schan=(5,), img=img, chdim=2)
        pairwise_gaussian = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])

        d = dcrf.DenseCRF2D(w, h, n_labels)
        d.setUnaryEnergy(U)
        d.addPairwiseEnergy(pairwise_gaussian, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        d.addPairwiseEnergy(pairwise_energy, compat=5, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)  # `compat` is the "strength" of this potential.

        Q = d.inference(50)
        pred_maps[i] = np.argmax(Q, axis=0).reshape((h,w))
    return pred_maps


def remove_smaller_components(im):
    if im.max()==0.0:
        return im
    sizes = {}
    im_ = im.copy()
    def dfs(i,j, root, key_elem, change_to):
        if i>=0 and i<im_.shape[0] and j>=0 and j< im_.shape[1] and im_[i,j] ==key_elem:
            im_[i][j]=change_to
            if root in sizes:
                sizes[root] += 1
            else:
                sizes[root] =0
            dfs(i-1,j,root,key_elem, change_to)
            dfs(i+1,j,root,key_elem, change_to)
            dfs(i,j-1,root,key_elem, change_to)
            dfs(i,j+1,root,key_elem, change_to)

    for i in range(im_.shape[0]):
        for j in range(im_.shape[1]):
            dfs(i,j, tuple((i,j)),1,2)

    big_comp = max(sizes, key=sizes.get)
    dfs(big_comp[0], big_comp[1], big_comp, 2,1)
    im_[im_>1] = 0
    return im_

def evaluate_test_folder(model, save_path=None, test_path=None, postproc=False, n_window=3):
    # Convert the data into input for the UNet
    img_path_list = [path for path in os.listdir(test_path + 'images/')]
    data = np.array([plt.imread(test_path + 'images/' + path) for path in img_path_list])
    annot = np.array([plt.imread(test_path + 'groundtruth/' + path) for path in img_path_list])
    #  print data.min(), data.max(), annot.max()
    n_frame = len(data)
    #  print(data.shape, annot.shape)
    if(data.shape[1]%16!=0 or data.shape[2]%16!=0):
        pad_width_h1 = int(np.floor((16-data.shape[1]%16)/2))
        pad_width_h2 = 16 - data.shape[1]%16 - pad_width_h1
        pad_width_w1 = int(np.floor((16-data.shape[2]%16)/2))
        pad_width_w2 = 16 - data.shape[2]%16 - pad_width_w1
        data = np.pad(data,((0,0),(pad_width_h1,pad_width_h2),(pad_width_w1,pad_width_w2)),'constant')
        annot = np.pad(annot,((0,0),(pad_width_h1,pad_width_h2),(pad_width_w1,pad_width_w2)),'constant')
    #  print(data.shape, annot.shape)
    #data_pad, annot_pad = zeropad(data, annot, h_max, w_max)
    if n_window==0:
        probas = model.predict(data[...,np.newaxis]*255., batch_size=8)[...,0]
    else:
        data_window = np.zeros((n_frame, data.shape[1], data.shape[2], n_window))
        n_window_half = int((n_window-1)/2)
        for i in range(n_window_half,n_frame-n_window_half):
            data_window[i] = np.rollaxis(data[i-n_window_half:i+n_window_half+1],0,3)
        #  print(data_window.shape)
        probas = model.predict(data_window*255.)[...,0]
    if postproc==True:
        probas = post_processing(data*255., probas)

    # Threshold predictions
    thresh = 0.5
    pred_maps = probas.copy()
    pred_maps[probas>=thresh] = 1#255
    pred_maps[probas<thresh] = 0

    #  for i in tqdm(range(pred_maps.shape[0])):
    #      pred_maps[i] = remove_smaller_components(pred_maps[i])

    # [h,w] = data[0].shape
    # dice_coef = [dice_coef_numpy(annot[i:i+1], pred_maps[i:i+1]) for i in range(n_frame)]
    # annot = [np.sum(annot[i:i+1])/(h*w*1.) for i in range(n_frame)]
    # annot_pred = [np.sum(pred_maps[i:i+1])/(h*w*1.) for i in range(n_frame)]

    #  top_50_indices = np.argpartition(np.array(annot_pred),-100)[-100:]
    #  annot_,dice_coef_,annot_pred_ = [],[],[]
    #  for i in top_50_indices.tolist():
    #      annot_.append(annot[i])
    #      annot_pred_.append(annot_pred[i])
    #      dice_coef_.append(dice_coef[i])


    # annot_,dice_coef_,annot_pred_ = [],[],[]
    # for i in range(n_frame):
        # if annot_pred[i]>0.04:
            # annot_.append(annot[i])
            # annot_pred_.append(annot_pred[i])
            # dice_coef_.append(dice_coef[i])
    # return  np.mean(np.array(dice_coef_))

    dice_coef_avg = 0.0
    for i in range(n_frame):
        dice_coef_avg += dice_coef_numpy(annot[i], pred_maps[i])
    dice_coef_avg /= n_frame
    print('Folder dice coef pred maps= ',dice_coef_avg)
    dice_coef = dice_coef_numpy(annot, probas)
    print('Folder dice coef = ',dice_coef)

    # Save the images onto disk
    if save_path !=None:
        for i in range(n_frame):
            plt.figure()
            ax = plt.subplot('131')
            ax.imshow(data[i], cmap='gray')
            ax.set_title('Actual Image')
            ax = plt.subplot('132')
            ax.imshow(annot[i], cmap='gray')
            ax.set_title('True Annotation')
            ax = plt.subplot('133')
            ax.imshow(pred_maps[i], cmap='gray')
            ax.set_title('Predicted Annotation')
            plt.savefig(save_path + img_path_list[i])
            plt.close()
    return dice_coef, pred_maps

