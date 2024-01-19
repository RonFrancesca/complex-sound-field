import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
import datetime
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True
print(torch.__version__)
from dataset import SoundFieldDataset

import utils
import sfun_torch as sfun
import ipdb

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    'font.size': 20})


def plot2pgf(temp, filename, folder='./'):
    """
    Converts numpy array to .txt file for plots using tikz
    :param temp: list of equally-long data
    :param filename: filename without extension nor path
    :param folder: folder where to save
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.savetxt(os.path.join(folder, filename + '.txt'), np.asarray(temp).T, fmt="%f", encoding='ascii')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config_2_l1_no_norm.json',
                        help='JSON-formatted file with configuration parameters')
    parser.add_argument('--best_model_path', default='ComplexUNet',
                        help='JSON-formatted file with configuration parameters')
    parser.add_argument('--mode', default='plot', help='if plotting or calculating')

    args = parser.parse_args()

    config_path = args.config
    # Load configuration
    if not os.path.exists(config_path):
        print('Error: No configuration file present at specified path.')
        return

    config = utils.load_config(config_path)
    print('Loaded configuration from: %s' % config_path)
    
    #ipdb.set_trace()
    base_dir = config["dataset"]["base_dir"]
    #print(base_dir)


    # Imports to select GPU
    os.environ['CUDA_ALLOW_GROWTH'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = config["run"]["gpu"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    cfg_dataset = config["dataset"]
    test_batch_size = 1
    do_normalize = config["dataset"]["do_normalize"]

    # load dataset
    if args.mode=='compute':
        for num_mics in [5, 15, 35, 55]:

            print('Computing for '+str(num_mics)+' microphones')
            sf_test = SoundFieldDataset(dataset_folder=cfg_dataset["test_path"], xSample=cfg_dataset["xSamples"],
                                        ySample=cfg_dataset["ySamples"], factor=cfg_dataset["factor"],do_test=True, num_mics=num_mics,do_normalize=do_normalize)

            test_loader = torch.utils.data.DataLoader(sf_test,
                                                      shuffle=False,
                                                      num_workers=0,  # torch.cuda.device_count() *4,  ## ??
                                                      batch_size=test_batch_size,
                                                      pin_memory=True)


            # inference time loop
            model = sfun.ComplexUnet(config["training"])
            model_name = config["training"]["session_id"]
            evaluation_folder_path = '/nas/home/lcomanducci/cxz/SR_ICASSP/complex-sound-field/models/'+str(model_name)
            evaluation_model_path = os.path.join(evaluation_folder_path, 'ComplexUNet')
            results_eval_path = os.path.join(evaluation_folder_path, 'results')
            os.makedirs(results_eval_path, exist_ok=True)
            model.load_state_dict(torch.load(evaluation_model_path, map_location='cuda:0'))
            model = model.to(device)

            results_dic = {
                'nmse': [],
                'nmse_complex': [],
                'ssim': []
            }
            frequencies = utils.get_frequencies()
            idx = 0
            for batch, (input_data, y_true) in enumerate(tqdm(test_loader)):
                model.eval()
                with torch.no_grad():
                    input_data = input_data.to(device)
                    y_true = y_true.to(device)
                    y_pred = model(input_data)
                    mask = input_data[:, 40:, :, :]

                    nmse = utils.NMSE_fun(y_pred[0].cpu().numpy(), y_true[0].cpu().numpy())
                    nmse_complex = utils.NMSE_complex_fun(y_pred[0].cpu().numpy(), y_true[0].cpu().numpy())

                    ssim_metric = utils.SSIM_fun(y_pred[0].cpu().numpy(), y_true[0].cpu().numpy())

                    results_dic['nmse'].append(nmse)
                    results_dic['nmse_complex'].append(nmse_complex)
                    results_dic['ssim'].append(ssim_metric)

                    do_plot = False
                    if do_plot:
                        n_freq = 10
                        plt.figure(figsize=(10, 5))
                        plt.subplot(131)
                        plt.title('Input')
                        plt.imshow(np.abs(input_data.cpu().numpy()[0, n_freq]),
                                   aspect='auto'), plt.colorbar(), plt.tight_layout()
                        plt.xlabel('$x [m]$'), plt.ylabel('$y [m] $'),
                        plt.subplot(132)
                        plt.title('Estimated')
                        plt.imshow(np.abs(y_pred.cpu().numpy()[0, n_freq]), aspect='auto'), plt.colorbar(), plt.tight_layout()
                        plt.xlabel('$x [m]$'), plt.ylabel('$y [m] $'),
                        plt.subplot(133)
                        plt.title('GT')
                        plt.imshow(np.abs(y_true.cpu().numpy()[0, n_freq]),
                                   aspect='auto'), plt.colorbar(), plt.tight_layout()
                        plt.xlabel('$x [m]$'), plt.ylabel('$y [m] $'),
                        plt.show()


                        n_freq = 5
                        plt.figure(figsize=(10, 5))
                        plt.subplot(131)
                        plt.title('Input')
                        plt.imshow(np.angle(input_data.cpu().numpy()[0, n_freq]),
                                   aspect='auto'), plt.colorbar(), plt.tight_layout()
                        plt.xlabel('$x [m]$'), plt.ylabel('$y [m] $'),
                        plt.subplot(132)
                        plt.title('Estimated')
                        plt.imshow(np.angle(y_pred.cpu().numpy()[0, n_freq]), aspect='auto'), plt.colorbar(), plt.tight_layout()
                        plt.xlabel('$x [m]$'), plt.ylabel('$y [m] $'),
                        plt.subplot(133)
                        plt.title('GT')
                        plt.imshow(np.angle(y_true.cpu().numpy()[0, n_freq]),
                                   aspect='auto'), plt.colorbar(), plt.tight_layout()
                        plt.xlabel('$x [m]$'), plt.ylabel('$y [m] $'),
                        plt.show()
                idx = idx +1
                if idx ==1000:
                    break



            average_nsme = 10 * np.log10(np.mean(results_dic['nmse'], axis=0))
            average_nsme_complex = 10 * np.log10(np.mean(results_dic['nmse_complex'], axis=0))

            # save nsme
            filename_path = os.path.join(results_eval_path, f'nmse_complex_{num_mics}.npy')
            np.save(filename_path, average_nsme, allow_pickle=False)

            # save complex nmse (ugly names)
            filename_path = os.path.join(results_eval_path, f'nmse_complex_COMPLEX_{num_mics}.npy')
            np.save(filename_path, average_nsme_complex, allow_pickle=False)

            average_ssim = np.mean(results_dic['ssim'], axis=0)
            # save nsme
            filename_path = os.path.join(results_eval_path, f'ssim_complex_{num_mics}.npy')
            np.save(filename_path, average_ssim, allow_pickle=False)

            x_values = frequencies

            tick_values = [30, 40, 50, 60, 70, 80, 90, 100, 200, 300]

            # calculate the NMSE
            plt.figure(figsize=(14, 10))
            plt.plot(x_values, average_nsme)
            plt.xscale('log')

            plt.xticks(tick_values, tick_values)

            plt.xlabel('$f [Hz]$'), plt.ylabel('$NMSE [dB]$')#, #plt.title('$\text{NMSE estimated from simulated data}$')
            plt.grid(which='both', linestyle='-', linewidth=0.5, color='gray')
            plt.show()

    else:
        T60 = 1
        model_name = config["training"]["session_id"]
        data_path = os.path.join(base_dir, 'models', str(model_name), 'results', f'n_mics/T60_{T60}')
        
        
        # nmse per number of microphone
        nmse_5_cxn = np.load(os.path.join(data_path,'nmse_complex_5.npy'))
        nmse_15_cxn = np.load(os.path.join(data_path,'nmse_complex_15.npy'))
        nmse_35_cxn = np.load(os.path.join(data_path,'nmse_complex_35.npy'))
        nmse_55_cxn = np.load(os.path.join(data_path,'nmse_complex_55.npy'))
        
        # lluis model
        # lluis_path ='/nas/home/lcomanducci/cxz/SR_ICASSP/sound-field-neural-network/sessions/session_4/simulated_data_evaluation/min_mics_5_max_mics_65_step_mics_10'
        # nmse_5_lluis = np.load(os.path.join(lluis_path,'nmse_lluis_5.npy'))
        # nmse_15_lluis = np.load(os.path.join(lluis_path,'nmse_lluis_15.npy'))
        # nmse_35_lluis = np.load(os.path.join(lluis_path,'nmse_lluis_35.npy'))
        # nmse_55_lluis = np.load(os.path.join(lluis_path,'nmse_lluis_55.npy'))

        # kernel based
        # kernel_path ='/nas/home/lcomanducci/cxz/SR_ICASSP/complex-sound-field/results_rebuttal'
        # nmse_5_kernel = np.load(os.path.join(kernel_path, 'nmse_sim_sig_proc_5_reg_0.1_.npy'))
        # nmse_15_kernel = np.load(os.path.join(kernel_path, 'nmse_sim_sig_proc_15_reg_0.1_.npy'))
        # nmse_35_kernel = np.load(os.path.join(kernel_path, 'nmse_sim_sig_proc_35_reg_0.1_.npy'))
        # nmse_55_kernel = np.load(os.path.join(kernel_path, 'nmse_sim_sig_proc_55_reg_0.1_.npy'))
        
        # nmse_5_kernel_complex = np.load(os.path.join(kernel_path, 'nmse_5_sig_proc_COMPLEX_reg_0.1.npy'))
        # nmse_15_kernel_complex = np.load(os.path.join(kernel_path, 'nmse_15_sig_proc_COMPLEX_reg_0.1_.npy'))
        # nmse_35_kernel_complex = np.load(os.path.join(kernel_path, 'nmse_35_sig_proc_COMPLEX_reg_0.1_.npy'))
        # nmse_55_kernel_complex = np.load(os.path.join(kernel_path, 'nmse_55_sig_proc_COMPLEX_reg_0.1_.npy'))


        tick_values = [30, 40, 50, 60, 70, 80, 90, 100, 200, 300]
        frequencies = utils.get_frequencies()

        # plot2pgf([frequencies, nmse_5_kernel], 'nmse_5_kernel', folder='results_rebuttal/pgfplots_001')
        # plot2pgf([frequencies, nmse_15_kernel], 'nmse_15_kernel', folder='results_rebuttal/pgfplots_001')
        # plot2pgf([frequencies, nmse_35_kernel], 'nmse_35_kernel', folder='results_rebuttal/pgfplots_001')
        # plot2pgf([frequencies, nmse_55_kernel], 'nmse_55_kernel', folder='results_rebuttal/pgfplots_001')

        # plot2pgf([frequencies, nmse_5_kernel_complex], 'nmse_complex_5_kernel', folder='results_rebuttal/pgfplots')
        # plot2pgf([frequencies, nmse_15_kernel_complex], 'nmse_complex_15_kernel', folder='results_rebuttal/pgfplots')
        # plot2pgf([frequencies, nmse_35_kernel_complex], 'nmse_complex_35_kernel', folder='results_rebuttal/pgfplots')
        # plot2pgf([frequencies, nmse_55_kernel_complex], 'nmse_complex_55_kernel', folder='results_rebuttal/pgfplots')

        x_values = frequencies

        # calculate the NMSE
        plt.figure(figsize=(14, 10))
        plt.plot(x_values, nmse_5_cxn,'-r')
        plt.plot(x_values, nmse_15_cxn,'-g')
        plt.plot(x_values, nmse_35_cxn,'-b')
        plt.plot(x_values, nmse_55_cxn,'-m')

        # plt.plot(x_values, 10*np.log10(np.mean(nmse_5_lluis,axis=0)),'--r')
        # plt.plot(x_values, 10*np.log10(np.mean(nmse_15_lluis,axis=0)),'--g')
        # plt.plot(x_values, 10*np.log10(np.mean(nmse_35_lluis,axis=0)),'--b')
        # plt.plot(x_values, 10*np.log10(np.mean(nmse_55_lluis,axis=0)),'--m')

        # plt.plot(x_values, nmse_5_kernel,'-*r')
        # plt.plot(x_values, nmse_15_kernel,'-*g')
        # plt.plot(x_values, nmse_35_kernel,'-*b')
        # plt.plot(x_values, nmse_55_kernel,'-*m')


        plt.xscale('log')
        plt.xticks(tick_values, tick_values)
        plt.xlabel('$f [Hz]$'), plt.ylabel('$NMSE [dB]$')#, plt.title('$\text{NMSE estimated from simulated data}$')
        plt.grid(which='both', linestyle='-', linewidth=0.5, color='gray')
        plt.legend(['5 $\mathrm{CxNet}$','15 $\mathrm{CxNet}$','35 $\mathrm{CxNet}$','55 $\mathrm{CxNet}$','5 $\mathrm{lluis}$','15 $\mathrm{lluis}$','35 $\mathrm{lluis}$','55 $\mathrm{lluis}$','5 $\mathrm{kernel}$','15 $\mathrm{kernel}$',])

        plt.show()
        
        plot_file_path = os.path.join(data_path, 'final_plot.png')
        plt.savefig(plot_file_path)
        #plt.savefig(os.path.join('/nas/home/lcomanducci/cxz/SR_ICASSP/complex-sound-field/plots','nmse_'+config["training"]["session_id"]+'.png'))
        do_save_plot = False

        if do_save_plot:
            plot2pgf([frequencies, nmse_5_cxn], 'nmse_5_cxn', folder='results')
            plot2pgf([frequencies, nmse_15_cxn], 'nmse_15_cxn', folder='results')
            plot2pgf([frequencies, nmse_35_cxn], 'nmse_35_cxn', folder='results')
            plot2pgf([frequencies, nmse_55_cxn], 'nmse_55_cxn', folder='results')

            """
            plot2pgf([frequencies, 10*np.log10(np.mean(nmse_5_lluis,axis=0))], 'nmse_5_lluis', folder='results')
            plot2pgf([frequencies, 10*np.log10(np.mean(nmse_15_lluis,axis=0))], 'nmse_15_lluis', folder='results')
            plot2pgf([frequencies, 10*np.log10(np.mean(nmse_35_lluis,axis=0))], 'nmse_35_lluis', folder='results')
            plot2pgf([frequencies, 10*np.log10(np.mean(nmse_55_lluis,axis=0))], 'nmse_55_lluis', folder='results')
            """
            # Do the same with complex NMSE
            model_name = config["training"]["session_id"]
            data_path ='/nas/home/lcomanducci/cxz/SR_ICASSP/complex-sound-field/models/'+str(model_name)+'/results'
            nmse_5_cxn = np.load(os.path.join(data_path,'nmse_complex_COMPLEX_5.npy'))
            nmse_15_cxn = np.load(os.path.join(data_path,'nmse_complex_COMPLEX_15.npy'))
            nmse_35_cxn = np.load(os.path.join(data_path,'nmse_complex_COMPLEX_35.npy'))
            nmse_55_cxn = np.load(os.path.join(data_path,'nmse_complex_COMPLEX_55.npy'))
            plot2pgf([frequencies, nmse_5_cxn], 'nmse_5_cxn_COMPLEX', folder='results')
            plot2pgf([frequencies, nmse_15_cxn], 'nmse_15_cxn_COMPLEX', folder='results')
            plot2pgf([frequencies, nmse_35_cxn], 'nmse_35_cxn_COMPLEX', folder='results')
            plot2pgf([frequencies, nmse_55_cxn], 'nmse_55_cxn_COMPLEX', folder='results')



        #plot2pgf([frequencies, 10*np.log10(np.mean(nmse_5_lluis,axis=0))], 'nmse_pwd_freq_db_missing_'+str(n_missing), folder=pgf_dataset_path)

        print('s')

if __name__ == '__main__':
    main()



