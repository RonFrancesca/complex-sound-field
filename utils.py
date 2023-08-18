import json
import numpy as np
import copy 
import ipdb
import torch
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

dir = '/nas/home/fronchini/complex-sound-field/figures'

def NMSE_fun(pred, gt):
    
    pred = pred.reshape(40, -1).cpu().numpy()
    gt = gt.reshape(40, -1).cpu().numpy()
    
    num = np.sum(np.power(np.abs(np.abs(gt) - np.abs(pred)),2),axis=-1)
    den = np.sum(np.power(np.abs(np.abs(gt)),2),axis=-1)
    
    nmse = num/den
    return nmse

def SSIM_fun(pred, gt):
    
    res = np.zeros((40, 1))
    
    pred = pred.reshape(40, -1).cpu().numpy() 
    gt = gt.reshape(40, -1).cpu().numpy()
    
    for freq in range(pred.shape[0]):
        data_range = np.abs(pred[freq, :]).max() - np.abs(pred[freq, :]).min()
        res[freq] = ssim(np.abs(pred[freq]), np.abs(gt[freq]), data_range=data_range)
    
    return res

   
 
def load_config(config_filepath):
    """ Load a session configuration from a JSON-formatted file.

    Args:
    config_filepath: string
    Returns: dict

    """

    try:
        config_file = open(config_filepath, 'r')
    except IOError:
        print('No readable config file at path: ' + config_filepath)
    else:
        with config_file:
            return json.load(config_file)

def get_frequencies():
    """Loads the frequency numbers found at 'util/frequencies.txt'.

    Returns: list

    """
    freqs_path = 'util/frequencies.txt'
    with open(freqs_path) as f:
        freqs = [[int(freq) for freq in line.strip().split(' ')] for line in f.readlines()][0]

    return freqs

def preprocessing(factor, sf, mask):
    """ Perfom all preprocessing steps.

        Args:
        factor: int
        sf: np.ndarray
        mask: np.ndarray

        Returns: np.ndarray, np.ndarray

        """

    # Downsampling
    downsampled_sf = downsampling(factor, sf)

    # Masking
    masked_sf = apply_mask(downsampled_sf, mask)

    # # Scaling masked sound field - we'll see thia later
    # scaled_sf = scale(masked_sf) # normalizzare parte reale ed immaginaria separate

    # Upsampling no-scaled sound field and mask
    irregular_sf, mask = upsampling(factor, masked_sf, mask) # irregular_sf shape (32, 32, 40), # mask shape (32, 32, 40
    
    
    return irregular_sf, mask


def downsampling(dw_factor, input_sfs):
    """ Downsamples sound fields given a downsampling factor.

        Args:
        dw_factor: int
        input_sfs: np.ndarray

        Returns: np.ndarray

        """
        
    
    return input_sfs[0:input_sfs.shape[0]:dw_factor, 0:input_sfs.shape[1]:dw_factor, :] 
    # we are consideing only one item per time, they are probably considering more than 1?


def apply_mask(input_sf, mask):
    """ Apply masks to sound fields.

        Args:
        input_sfs: np.ndarray
        masks: np.ndarray

        Returns: np.ndarray

        """
    
    #masked_sfs = []
    #for sf, mk in zip(input_sfs, masks):
    
    aux_sf = copy.deepcopy(input_sf)
    aux_sf[mask==0] = 0
    
    for i in range(input_sf.shape[2]):
        #aux_max = aux_sf[:, :, i].max()
        aux_max = 0 
        input_sf[:, :, i][mask[:, :, i]==0] = aux_max
        #masked_sfs.append(sf)

    return input_sf

def upsampling(up_factor, sf, mask):
    """ Upsamples sound fields and masks given a upsampling factor.

        Args:
        up_factor: int
        input_sfs: np.ndarray
        masks: np.ndarray

        Returns: np.ndarray, np.ndarray

        """

    
    sf_up = []
    
    sf = np.swapaxes(sf, 2, 0)
    mask = np.swapaxes(mask, 2, 0)
    
    for sf_slice in sf:
        positions = np.repeat(range(1, sf_slice.shape[1]), up_factor-1) #positions in sf slice to put 1
        sf_slice_up = np.insert(sf_slice, obj=positions, values=np.zeros(len(positions)), axis=1) 
        sf_slice_up = np.transpose(np.insert(np.transpose(sf_slice_up),obj=positions,values=np.zeros(len(positions)), axis=1)) 
        sf_slice_up = np.pad(sf_slice_up, (0,up_factor-1),  mode='constant', constant_values=0) 
        sf_slice_up = np.roll(sf_slice_up, (up_factor-1)//2, axis=0)
        sf_slice_up = np.roll(sf_slice_up, (up_factor-1)//2, axis=1)
        sf_up.append(sf_slice_up) #len(sf_up) = 40, sf_slice_up shape [32, 32]

    mask_slice = mask[0, :, :]
    positions = np.repeat(range(1, mask_slice.shape[1]), up_factor-1) #positions in mask slice to put 0
    mask_slice_up = np.insert(mask_slice, obj=positions,values=np.zeros(len(positions)), axis=1)
    mask_slice_up = np.transpose(np.insert(np.transpose(mask_slice_up),obj=positions,values=np.zeros(len(positions)), axis=1))
    mask_slice_up = np.pad(mask_slice_up, (0,up_factor-1),  mode='constant')
    mask_slice_up = np.roll(mask_slice_up, (up_factor-1)//2, axis=0)
    mask_slice_up = np.roll(mask_slice_up, (up_factor-1)//2, axis=1)
    mask_slice_up = mask_slice_up[np.newaxis, :]
    mask_up = np.repeat(mask_slice_up, mask.shape[0], axis=0) #len(sf_up) = 40, sf_slice_up shape [32, 32]


        # batch_sf_up.append(sf_up)
        # batch_mask_up.append(mask_up)

    sf_up = np.asarray(sf_up)
    sf_up = np.swapaxes(sf_up, 2, 0)

    mask_up = np.asarray(mask_up)
    mask_up = np.swapaxes(mask_up, 2, 0)
    

    return sf_up, mask_up