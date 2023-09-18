from torch.utils.data import Dataset
import os
import numpy as np
import torchaudio
import random
import torch
import glob
import scipy
import copy


import utils

def norm_sf_complex(input_ten):
    gt_real_norm = (input_ten.real - input_ten.real.mean()) / input_ten.real.std()
    gt_imag_norm = (input_ten.imag - input_ten.imag.mean()) / input_ten.imag.std()
    norm_ten = gt_real_norm.type(torch.complex64) + 1j * gt_imag_norm.type(torch.complex64)
    return norm_ten


def generate_mask(height, width, channels, num_mics=10):
    
    mask_slice = np.zeros((height*width), np.uint8) ### ???? if everything is cast to complex or everything is set to 0, need to multiply or not

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
        dataset_folder=None,
        set_file_list=None,
        xSample=32, 
        ySample=32, 
        factor=4,
        return_dims = False,
        do_normalize = True,
        num_mics = 10,
        do_test = False,
        do_plot = False
        
        
    ):
        
        self.dataset_folder = dataset_folder
        self.freq = utils.get_frequencies()
        self.num_freq = len(self.freq)
        self.return_dims = return_dims
        self.do_normalize = do_normalize
        self.do_test = do_test
        self.num_mics = num_mics
        self.do_plot = do_plot
        
        if dataset_folder is None and set_file_list is None:
            print(f"Only one of those can be sett to None")
            #TODO: Throw error
            return
            
        if dataset_folder is None:
            self.soundfield_list = set_file_list
        else:
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
        

        f_response_complex = np.transpose(f_response_complex, (1, 0, 2))

        sf_gt= f_response_complex[:, :, frequencies] # considering only 40 frequencies [32, 32, 40]

        initial_sf = copy.deepcopy(sf_gt)
        

        # Get mask samples (always the same mask so far)
        if self.do_test:
            if self.num_mics is None:
                print('Error: No number of microphones provided, we are in test mode!.')
                return
            mask = generate_mask(int(self.xSample / self.factor), int(self.ySamples / self.factor), self.num_freq, self.num_mics)
        else:
            num_mics_list = [5, 15, 35, 55]
            num_mics = random.choice(num_mics_list)
            mask = generate_mask(int(self.xSample/self.factor), int(self.ySamples/self.factor), self.num_freq, num_mics)

        # # preprocessing
        mask_downsampled = mask

        irregular_sf, mask = utils.preprocessing(self.factor, initial_sf, mask)

        irregular_sf = torch.from_numpy(irregular_sf)
        mask = torch.from_numpy(mask)

        if self.do_normalize:
            irregular_sf = norm_sf_complex(irregular_sf)

        sf_masked = torch.cat((irregular_sf, mask), dim=2) 
        
        sf_masked = torch.moveaxis(sf_masked, 2, 0) 
        sf_gt = torch.moveaxis(torch.from_numpy(sf_gt), 2, 0)
        if self.do_normalize:
            sf_gt = norm_sf_complex(sf_gt)

        # Return also room dimensions for plots
        x_dim = float(self.soundfield_list[item].split('_')[3])
        y_dim = float(self.soundfield_list[item].split('_')[4])

        if self.return_dims:
            if self.do_plot:
                return sf_masked, sf_gt,mask_downsampled, x_dim, y_dim
            else:
                return sf_masked, sf_gt, x_dim, y_dim
        else:
            if self.do_plot:
                return sf_masked, sf_gt, mask_downsampled
            else:
                return sf_masked, sf_gt
