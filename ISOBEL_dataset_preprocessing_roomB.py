import os
import scipy
import matplotlib.pyplot as plt
import numpy as np
import resampy
from tqdm import tqdm
import matplotlib.pyplot as plt
rate_isobel = 48000
rate_lluis = 1200
type ='rir' # 'sf'
import torch
import scipy
import os
PLOT = False # Plot example of sound field
BASE_DIR = '/nas/home/lcomanducci/cxz/EUSIPCO/complex-sound-field/dataset/test/RoomB'
print('Processing ISOBEL dataset Room B')
source_names = ['source_1', 'source_2']
for s in tqdm(source_names):
    if type == 'sf':
        base_path = '/nas/public/dataset/ISOBEL_SF_Dataset/Room B/RoomB_SoundField_Raw/{}/h_100'.format(s)
        rir_length = 197000
        rir_length_resampled = 4925

    if type == 'rir':
        base_path = '/nas/public/dataset/ISOBEL_SF_Dataset/Room B/RoomB_SoundField_IRs/{}/h_100'.format(s)
        rir_length = 65536
        rir_length_resampled = 1639

    x_dim, y_dim = 32, 32
    room_response_matrix = np.zeros((x_dim, y_dim, rir_length),dtype=np.float64)
    room_response_matrix_resampled = np.zeros((x_dim, y_dim, rir_length_resampled),dtype=np.float64)
    # Load time domain signals from Isobel Dataset
    for x_idx in tqdm(range(x_dim)):
        for y_idx in range(y_dim):
            curr_file = os.path.join(base_path, 'idxX_{}_idxY_{}.mat'.format(x_idx+1, y_idx+1)) # N.B. here ugly indexing, but it is NOT A TYPO, shitch X and Y axes due to matrix-handling of data
            if type == 'sf':
                room_response_matrix[x_idx,y_idx] = scipy.io.loadmat(curr_file)['RawResponse'][:, 0]
            if type == 'rir':
                room_response_matrix[x_idx,y_idx] = normalize(scipy.io.loadmat(curr_file)['ImpulseResponse'][:, 0])


    # Resample time domain signals from Isobel Dataset to do similarly to Lluis et al.
    for x_idx in tqdm(range(x_dim)):
        for y_idx in range(y_dim):
            room_response_matrix_resampled[x_idx, y_idx] = scipy.signal.decimate(room_response_matrix[x_idx, y_idx], 40)

    Room_rtf_matrix = np.zeros((32, 32, room_response_matrix_resampled.shape[-1]//2 +1), dtype=np.complex128)
    Room_rtf_matrix = np.fft.rfft(room_response_matrix_resampled, axis=-1)
    n = room_response_matrix_resampled.shape[-1]
    fft_axis = np.fft.rfftfreq(n, d=1./rate_lluis)

    # N.B. lluis considers only <= 300 Hz
    for i in range(len(fft_axis)):
        if fft_axis[i] > 300:
            break
    idx_300 = i-1

    # Now let's check which freqs in the axis are closer to the ones defined by Lluis
    lluis_frequencies=np.arange(0,600)
    fft_map_idx = np.zeros_like(lluis_frequencies)
    for i in range(len(lluis_frequencies)):
        fft_map_idx[i] = np.argmin(np.abs(lluis_frequencies[i]-fft_axis))

    # Now we have the Room response at the corresponding frequencies
    Room_rtf_matrix_lluis = Room_rtf_matrix[:,:,fft_map_idx.astype(int)]
    fft_axis_lluis = fft_axis[fft_map_idx.astype(int)]

    if PLOT:
        f_idx = 42
        plt.figure()
        plt.subplot(121)
        plt.imshow((np.abs(Room_rtf_matrix_lluis[:,:,f_idx]).T),aspect='auto'), plt.colorbar()
        plt.gca().invert_yaxis()
        plt.xlabel('x [m]'), plt.ylabel('y [m]')
        plt.title('Freq: {} Hz Absolute Value'.format(np.round(fft_axis_lluis[f_idx],2)))
        plt.subplot(122)
        plt.imshow(np.angle(Room_rtf_matrix_lluis[:, :, f_idx]).T,aspect='auto'), plt.colorbar()
        plt.xlabel('x [m]'), plt.ylabel('y [m]')
        plt.title(' Phase'.format(fft_axis_lluis[f_idx]))
        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.show()


    filename = ('roomB_d_4.16_6.46_2.30_s_{}_.mat'.format(s))
    mdic = {'AbsFrequencyResponse': np.abs(Room_rtf_matrix_lluis), 'FrequencyResponse': Room_rtf_matrix_lluis}
    scipy.io.savemat(os.path.join(BASE_DIR, filename), mdic)
