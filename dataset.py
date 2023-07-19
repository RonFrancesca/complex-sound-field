from torch.utils.data import Dataset
import os
import numpy as np
import torchaudio
import random
import torch
import glob
import scipy
from pathlib import Path
import ipdb
import copy


import utils


def generate_mask(height, width, channels, num_mics=6):
    
    mask_slice = np.zeros((height*width), np.uint8) ### ???? if everything is cast to complex or everything is set to 0, need to multiply or not
    
    # mask_slice = torch.tensor(np.zeros((height*width), np.uint8)) # ??
    # if elementwise element do not change -> ok esempio Luca is ok
    
    # test

    num_holes = height*width - num_mics

    index_holes = np.random.choice(int(height*width), size=num_holes, replace=False)

    mask_slice[index_holes] = 1

    mask_slice.resize(height, width, 1)
        
    mask_slice = 1-mask_slice

    mask = np.repeat(mask_slice, channels, axis=2)

    return mask

class SoundFieldDataset(Dataset):
    def __init__(
        self,
        dataset_folder,
        xSample=32, 
        ySample=32, 
        factor=4,
        
        
    ):
        
        self.dataset_folder = dataset_folder
        self.freq = utils.get_frequencies()
        self.num_freq = len(self.freq)
        self.soundfield_list = glob.glob(os.path.join(self.dataset_folder, "*.mat"))
        self.xSample = int(xSample)
        self.ySamples = int(ySample)
        self.factor = int(factor) 
        
        
    def __len__(self):
        return len(self.soundfield_list)

    def __getitem__(self, item):
        sf_item = self.soundfield_list[item]

        frequencies = np.asarray(self.freq)
        mat = scipy.io.loadmat(sf_item)
        
        f_response_complex = mat['FrequencyResponse'].astype(np.complex64)
        
        f_response_complex = np.transpose(f_response_complex, (1, 0, 2)) # transpose (x, y room) -> plot
        # plot
        
        sf_gt= f_response_complex[:, :, frequencies] # considering only 40 frequencies [32, 32, 40]
        
        
        # as far as I get , they take the sample as gt, the pre-process is the label itself and I have no idea on how to pre-process the dataset
        initial_sf = copy.deepcopy(sf_gt)
        

        # Get mask samples (always the same mask so far)
        mask = generate_mask(int(self.xSample/self.factor), int(self.ySamples/self.factor), self.num_freq)
        
        # # preprocessing
        irregular_sf, mask = utils.preprocessing(self.factor, initial_sf, mask)
        sf_masked = np.concatenate((irregular_sf, mask), axis=2) 
        
        sf_masked = np.moveaxis(sf_masked, 2, 0) 
        sf_gt = np.moveaxis(sf_gt, 2, 0)

        # Scale ground truth sound field (we'll see it later)
        #sf_gt = util.scale(sf_gt)
        
        #ipdb.set_trace()
        # sf_masked = torch.tensor(sf_masked, dtype=torch.complex64) ##TODO: Could be this one? 
        return sf_masked, sf_gt # shapes: [32, 32, 80], [32, 32, 40]
        #return [irregular_sf, mask], sf_gt
    
        