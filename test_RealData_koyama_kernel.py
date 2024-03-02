import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import numpy as np
import scipy.special as special
import scipy.spatial.distance as distfuncs
import utils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True
print(torch.__version__)
from dataset import SoundFieldDataset
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
test_path_luca = '/nas/home/lcomanducci/cxz/EUSIPCO/complex-sound-field/dataset/test/RoomB'  # Luca
ITERATIONS = 100
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    'font.size': 20})

def plot2pgf(temp, filename, folder='./'):
    """
    :param temp: list of equally-long data
    :param filename: filename without extension nor path
    :param folder: folder where to save
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.savetxt(os.path.join(folder, filename + '.txt'), np.asarray(temp).T, fmt="%f", encoding='ascii')


def kiFilterGen(k, posMic, posEst, filterLen=None, smplShift=None, reg=1e-1):
    """Kernel interpolation filter for estimating pressure distribution from measurements
    - N. Ueno, S. Koyama, and H. Saruwatari, “Kernel Ridge Regression With Constraint of Helmholtz Equation
      for Sound Field Interpolation,” Proc. IWAENC, DOI: 10.1109/IWAENC.2018.8521334, 2018.
    - N. Ueno, S. Koyama, and H. Saruwatari, “Sound Field Recording Using Distributed Microphones Based on
      Harmonic Analysis of Infinite Order,” IEEE SPL, DOI: 10.1109/LSP.2017.2775242, 2018.
    """
    numMic = posMic.shape[0]
    numEst = posEst.shape[0]
    numFreq = k.shape[0]
    fftlen = numFreq*2

    if filterLen is None:
        filterLen = numFreq+1
    if smplShift is None:
        smplShift = numFreq/2

    k = k[:, None, None]
    distMat = distfuncs.cdist(posMic, posMic)[None, :, :]
    K = special.spherical_jn(0, k * distMat)
    eigK, _ = np.linalg.eig(K)
    regK =  eigK[:,0] * reg
    Kinv = np.linalg.inv(K + regK[:,None,None] * np.eye(numMic)[None, :, :])
    distVec = np.transpose(distfuncs.cdist(posEst, posMic), (1, 0))[None, :, :]
    kappa = special.spherical_jn(0, k * distVec)
    kiTF = np.transpose(kappa, (0, 2, 1)) @ Kinv
    kiTF = np.concatenate((np.zeros((1, numEst, numMic)), kiTF, kiTF[int(fftlen/2)-2::-1, :, :].conj()))
    kiFilter = np.fft.ifft(kiTF, n=fftlen, axis=0).real
    kiFilter = np.concatenate((kiFilter[fftlen-smplShift:fftlen, :, :], kiFilter[:filterLen-smplShift, :, :]))
    return kiFilter


BASE_DIR = '/nas/home/fronchini/complex-sound-field'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.json',
                        help='JSON-formatted file with configuration parameters')
    parser.add_argument('--reg', default=0.01, help='reg param',type=float)
    parser.add_argument('--num_mics', default=64, help='reg param',type=int)
    parser.add_argument('--t60', default=0.4, help='Reverberation time [s]',type=float)

    args = parser.parse_args()
    config_path = args.config
    config = utils.load_config(config_path)

    xSamples = 32
    ySamples = 32
    factor = 4 # Useless here
    if args.t60 == 1.0:
        args.t60 = int(args.t60)
    cfg_dataset_test = os.path.join(BASE_DIR,'dataset/test','T60_'+str(args.t60))
    test_batch_size = 1
    num_mics = args.num_mics

    for num_mics in config["dataset"]["num_mics_list"]:

        print('Computing for Number of Microphones: '+str(num_mics)
              +', Reverberation time:' +str(args.t60)+
              '[s], Regularization parameter: '+str(args.reg))

        sf_test = SoundFieldDataset(dataset_folder=test_path_luca, xSample=xSamples,
                                    ySample=ySamples, factor=factor,return_dims=True,do_normalize=False, num_mics=num_mics,do_test=True)

        test_loader = torch.utils.data.DataLoader(sf_test,
                                                  shuffle=False,
                                                  num_workers=4,  # 4,   ## ??
                                                  batch_size=test_batch_size)  ##2 number of workers

        results_dic = {
            'nmse': [],
            'nmse_complex': [],
            'ssim': []
        }

        frequencies = utils.get_frequencies()

        # Params Stuff
        # Filter parameters
        # Kernel interpolation filter
        c = 343
        f_s = 1200  # sampling freq cfr https://github.com/RonFrancesca/complex-sound-field/blob/main/create_dataset/init.m#L8
        fftlen = f_s
        k = 2 * np.pi * np.array(frequencies) / c
        reg = args.reg
        do_plot = False
        for iterations in tqdm(range(ITERATIONS)):
            for batch, (input_data, y_true, x_dim, y_dim) in enumerate(test_loader):
                mask = input_data[:, 40:, :, :]


                # N.B. still messy
                if do_plot:
                    mask_down = np.load('/nas/home/lcomanducci/cxz/SR_ICASSP/complex-sound-field/plots_soundfield/mask_lluis_15.npy')
                    mask_down = np.load('/nas/home/lcomanducci/cxz/SR_ICASSP/complex-sound-field/plots_soundfield/mask_lluis_'+str(num_mics)+'_higher_freq.npy')
                    irregular_sf, mask = utils.preprocessing(4, np.random.randn(32,32,40), mask_down[0])
                    mask = np.expand_dims(np.transpose(mask,(2,0,1)),axis=0)


                """
                Kernel interpolation for soundfield reconstruction
                """
                # Params related to room
                d_x, d_y = x_dim/mask.shape[2], y_dim/mask.shape[3]
                idx_mic_x, idx_mic_y = np.where(np.real(mask[0,0]))
                mic_pos_x, mic_pos_y = d_x * idx_mic_x, d_y * idx_mic_y
                posMic = np.array([mic_pos_x.numpy(),mic_pos_y.numpy()]).T
                x_mic_eval,y_mic_eval = np.meshgrid(np.arange(mask.shape[2])*(x_dim/mask.shape[2]).numpy(), np.arange(mask.shape[3])*(y_dim/mask.shape[3]).numpy())
                posEval = np.array([x_mic_eval.ravel(), y_mic_eval.ravel()]).T
                # Kernel filter compute
                kiFilter = kiFilterGen(k, posMic, posEval,  filterLen=len(k), smplShift=len(k)*2,reg=reg)
                input_data = input_data.to(device)
                y_true = y_true.to(device)
                kiFilter = torch.Tensor(kiFilter).to(device)
                kiFilterSpec = torch.fft.rfft(kiFilter,n=fftlen,dim=0)[frequencies]
                SigMicSpec = input_data[0,:40,idx_mic_x,idx_mic_y].unsqueeze(-1)
                SigEst = torch.matmul(kiFilterSpec,SigMicSpec).squeeze(0)
                SigEst = torch.reshape(SigEst,(len(frequencies), mask.shape[2], mask.shape[3]))
                SigEst = torch.permute(SigEst,dims=[0,2,1])

                """
                End of Kernel interpolation
                """
                # Change device to compute results
                y_true = y_true[0].cpu().numpy()

                if do_plot:
                    n_freq = 5
                    plt.figure(figsize=(10,5))
                    plt.subplot(131)
                    plt.title('Input')
                    plt.imshow(np.abs(input_data.cpu().numpy()[0, n_freq]), aspect='auto'),plt.colorbar(),plt.tight_layout()
                    plt.xlabel('$x [m]$'), plt.ylabel('$y [m] $'),
                    plt.subplot(132)
                    plt.title('Estimated')
                    plt.imshow(np.abs(SigEst.cpu()[ n_freq]), aspect='auto'),plt.colorbar(),plt.tight_layout()
                    plt.xlabel('$x [m]$'), plt.ylabel('$y [m] $'),
                    plt.subplot(133)
                    plt.title('GT')
                    plt.imshow(np.abs(y_true[ n_freq]), aspect='auto'),plt.colorbar(),plt.tight_layout()
                    plt.xlabel('$x [m]$'), plt.ylabel('$y [m] $'),
                    plt.show()

                    x_dim, y_dim = x_dim.numpy()[0], y_dim.numpy()[0]
                    plt.figure(figsize=(5, 5))
                    plt.imshow(np.abs(SigEst.cpu().numpy()[ n_freq]), aspect='auto', extent=[0, x_dim, 0, y_dim],
                               cmap='magma'), plt.tight_layout()
                    plt.xlabel('$x [m]$'), plt.ylabel('$y [m] $')
                    plt.savefig('plots_soundfield/abs_kernel_est_' + str(frequencies[n_freq]) + '_num_mics' + str(num_mics) + '.pdf')
                    plt.show()
                    plt.figure(figsize=(5, 5))
                    plt.imshow(np.angle(SigEst.cpu().numpy()[ n_freq]), aspect='auto', extent=[0, x_dim, 0, y_dim],
                               cmap='magma'), plt.tight_layout()
                    plt.xlabel('$x [m]$'), plt.ylabel('$y [m] $')
                    plt.savefig('plots_soundfield/phase_kernel_est_' + str(frequencies[n_freq]) + '_num_mics' + str(num_mics) + '.pdf')
                    plt.show()


                    plt.figure(figsize=(5, 5))
                    plt.imshow(np.abs(y_true.cpu().numpy()[0,n_freq]), aspect='auto', extent=[0, x_dim, 0, y_dim],
                               cmap='magma'), plt.tight_layout()
                    plt.xlabel('$x [m]$'), plt.ylabel('$y [m] $')
                    #plt.savefig('plots_soundfield/phase_gt_' + str(frequencies[n_freq]) + '_num_mics' + str(num_mics) + '.pdf')
                    plt.show()



                # Compute Metrics
                nmse = utils.NMSE_fun(SigEst.cpu().numpy(), y_true)
                ssim_metric = utils.SSIM_fun(SigEst.cpu().numpy(), y_true)
                nmse_complex = utils.NMSE_complex_fun(SigEst.cpu().numpy(),y_true)

                # Append metrics to dictionary
                results_dic['nmse'].append(nmse)
                results_dic['ssim'].append(ssim_metric)
                results_dic['nmse_complex'].append(nmse_complex)

        # Do averages
        average_nsme = 10 * np.log10(np.mean(results_dic['nmse'], axis=0))
        average_nsme_complex = 10 * np.log10(np.mean(results_dic['nmse_complex'], axis=0))
        average_ssim = np.mean(results_dic['ssim'], axis=0)

        results_eval_path = '/nas/home/lcomanducci/cxz/EUSIPCO/complex-sound-field/models/eusipco_large_32/results/real/n_mics/test_set'
        # Save numpy array and PgfPlot
        np.save(os.path.join(results_eval_path,'nmse_sim_sig_proc_abs_'+str(num_mics)+'_reg_'+str(reg)+'_.npy'),average_nsme)
        np.save(os.path.join(results_eval_path,'mssim_sim_sig_proc_'+str(num_mics)+'_reg_'+str(reg)+'_.npy'),average_ssim)
        np.save(os.path.join(results_eval_path,'nmse_sim_sig_proc_complex_'+str(num_mics)+'_reg_'+str(reg)+'_.npy'),average_nsme_complex)


if __name__ == '__main__':
    main()




