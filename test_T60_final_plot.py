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
        calcolate = 'complex'
        
        for num_mics in [5, 15, 35, 55]:
            
            model_name = config["training"]["session_id"]
            data_path = os.path.join(base_dir, 'models', str(model_name), 'results', 'T60', f"n_mics_{num_mics}")
            
            # path to kernel method and lluis method
            kernel_path ='/nas/home/lcomanducci/cxz/SR_ICASSP/complex-sound-field/results_eusipco'
            lluis_path = '/nas/home/fronchini/sound-field-neural-network/sessions/session_04-16-bs4/simulated_data_evaluation/min_mics_5_max_mics_65_step_mics_5'
            
            tick_values = [30, 40, 50, 60, 70, 80, 90, 100, 200, 300]
            frequencies = utils.get_frequencies()
            x_values = frequencies
            
            # T60 considered
            T60_list = [0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]
            
            # average_data_our_network = []
            # average_data_lluis = []
            # average_data_kernel_based = []
            
            plt.figure(figsize=(14, 10))
        
            # colors considered for the three models
            colors = ['-c', '-m', '-y', '-k', '-g', '-b', '-r']
            colors_kernel = ['-*c', '-*m', '-*y', '-*k', '-*g', '-*b', '-*r']
            colors_lluis = ['--c', '--m', '--y', '--k', '--g', '--b', '--r']
            
            
            if calcolate == 'magnitude':
                
                plot_filename = f'results_nmics_{num_mics}.pdf'
                
                for idx, T60 in enumerate(T60_list):
                    
                    # our network
                    nsme_T60_tmp = np.load(os.path.join(data_path, f'nmse_complex_{T60}.npy'))
                    if T60 == 1:
                        label = 'T60 1.0'
                    else:
                        label = f'T60 {T60}'
                    plt.plot(x_values, nsme_T60_tmp, colors[idx], label=label)
                    #average_data_our_network.append(nsme_T60_tmp)
                    
                    # lluis network
                    lluis_path_tmp = os.path.join(lluis_path, f'T60_{T60}')
                    nmse_T60_lluis_tmp = np.load(os.path.join(lluis_path_tmp, f'nmse_lluis_{num_mics}.npy'))
                    plt.plot(x_values, 10*np.log10(np.mean(nmse_T60_lluis_tmp,axis=0)), colors_lluis[idx], label=None)
                    #average_data_lluis.append(10*np.log10(np.mean(nmse_T60_lluis_tmp,axis=0)))
                    
                    # kernel based
                    nsme_T60_kernel_tmp = np.load(os.path.join(kernel_path, f'nmse_sim_sig_proc_{num_mics}_mics_reg_0.1_t60_{T60}_.npy'))
                    plt.plot(x_values, nsme_T60_kernel_tmp, colors_kernel[idx], label=None)
                    #average_data_kernel_based.append(nsme_T60_kernel_tmp)
                    
                plt.legend()
            
            elif calcolate == 'complex':
                # create a new figure for each T60
                
                plot_filename = f'results_complex_nmics_{num_mics}.pdf'
                
                for idx, T60 in enumerate(T60_list):
                    
                    if T60 == 1:
                        label = 'T60 1.0'
                    else:
                        label = f'T60 {T60}'
                        
                    # our network
                    nsme_T60_tmp = np.load(os.path.join(data_path,f'nmse_complex_COMPLEX_{T60}.npy'))
                    plt.plot(x_values, nsme_T60_tmp, colors[idx], label=label)
                    
                    # kernel based
                    nsme_T60_kernel_tmp = np.load(os.path.join(kernel_path, f'nmse_sim_sig_proc_COMPLEX{num_mics}_mics_reg_0.1_t60_{T60}_.npy'))
                    plt.plot(x_values, nsme_T60_kernel_tmp, colors_kernel[idx], label=None)
                
                plt.legend()
            
            plt.xscale('log')
            plt.xticks(tick_values, tick_values)
            plt.xlabel('$f [Hz]$'), plt.ylabel('$NMSE [dB]$')#, plt.title('$\text{NMSE estimated from simulated data}$')
            plt.grid(which='both', linestyle='-', linewidth=0.5, color='gray')  
            plt.show()
            plot_file_path = os.path.join(data_path, plot_filename)
            plt.savefig(plot_file_path)
            plt.close() 
            
            
            # average and variance proposed method
            # average_ronchini = np.mean(average_data_our_network, axis=0) 
            # variance_our_network = np.std(average_data_our_network, axis=0)
            # se_ronchini = np.std(average_data_our_network, axis=0) / np.sqrt(len(average_data_our_network))
            
            # average and variance lluis network
            # average_lluis = np.mean(average_data_lluis, axis=0) 
            # variance_lluis = np.std(average_data_lluis, axis=0)
            # se_lluis = np.std(average_data_lluis, axis=0) / np.sqrt(len(average_data_lluis))
            
            # average and variance kernel based network
            # average_kernel_based = np.mean(average_data_kernel_based, axis=0) 
            # variance_kernel_based = np.std(average_data_kernel_based, axis=0)
            # se_kernel_based = np.std(average_data_kernel_based, axis=0) / np.sqrt(len(average_data_kernel_based))
            
            
            # Plot average and variance
            
            # plt.figure(figsize=(14, 10))
            # alpha = 0.5
            # plt.plot(x_values, average_ronchini, color='b', label='Average')
            # #plt.fill_between(x_values, average_ronchini - variance_our_network, average_ronchini + variance_our_network, color='b', alpha=alpha, label='Std')
            
            # plt.plot(x_values, average_lluis, color='orange', label='Average')
            # #plt.fill_between(x_values, average_lluis - variance_lluis, average_lluis + variance_lluis, color='orange', alpha=alpha, label='Std')
            
            # plt.plot(x_values, average_kernel_based, color='g', label='Average')
            # #plt.fill_between(x_values, average_kernel_based - variance_kernel_based, average_kernel_based + variance_kernel_based, color='g', alpha=alpha, label='Std')
            
            # plt.legend()
            # plt.xlabel('$f [Hz]$')
            # plt.ylabel('$NMSE [dB]$')
            # plt.xscale('log')
            # plt.xticks(tick_values, tick_values)
            # plt.grid(which='both', linestyle='-', linewidth=0.5, color='gray')
            # plot_file_path = os.path.join(data_path, f'final_plot_{calcolate}_average_std.png')
            # plt.savefig(plot_file_path)
            
            # plt.show()
            
            # plot2pgf([frequencies, nmse_5_kernel], 'nmse_5_kernel', folder='results_rebuttal/pgfplots_001')
            # plot2pgf([frequencies, nmse_15_kernel], 'nmse_15_kernel', folder='results_rebuttal/pgfplots_001')
            # plot2pgf([frequencies, nmse_35_kernel], 'nmse_35_kernel', folder='results_rebuttal/pgfplots_001')
            # plot2pgf([frequencies, nmse_55_kernel], 'nmse_55_kernel', folder='results_rebuttal/pgfplots_001')

            # plot2pgf([frequencies, nmse_5_kernel_complex], 'nmse_complex_5_kernel', folder='results_rebuttal/pgfplots')
            # plot2pgf([frequencies, nmse_15_kernel_complex], 'nmse_complex_15_kernel', folder='results_rebuttal/pgfplots')
            # plot2pgf([frequencies, nmse_35_kernel_complex], 'nmse_complex_35_kernel', folder='results_rebuttal/pgfplots')
            # plot2pgf([frequencies, nmse_55_kernel_complex], 'nmse_complex_55_kernel', folder='results_rebuttal/pgfplots')            
            
            # do_save_plot = False

            # if do_save_plot:
            #     plot2pgf([frequencies, nmse_5_cxn], 'nmse_5_cxn', folder='results')
            #     plot2pgf([frequencies, nmse_15_cxn], 'nmse_15_cxn', folder='results')
            #     plot2pgf([frequencies, nmse_35_cxn], 'nmse_35_cxn', folder='results')
            #     plot2pgf([frequencies, nmse_55_cxn], 'nmse_55_cxn', folder='results')

            #     """
            #     plot2pgf([frequencies, 10*np.log10(np.mean(nmse_5_lluis,axis=0))], 'nmse_5_lluis', folder='results')
            #     plot2pgf([frequencies, 10*np.log10(np.mean(nmse_15_lluis,axis=0))], 'nmse_15_lluis', folder='results')
            #     plot2pgf([frequencies, 10*np.log10(np.mean(nmse_35_lluis,axis=0))], 'nmse_35_lluis', folder='results')
            #     plot2pgf([frequencies, 10*np.log10(np.mean(nmse_55_lluis,axis=0))], 'nmse_55_lluis', folder='results')
            #     """
            #     # Do the same with complex NMSE
            #     model_name = config["training"]["session_id"]
            #     data_path ='/nas/home/lcomanducci/cxz/SR_ICASSP/complex-sound-field/models/'+str(model_name)+'/results'
            #     nmse_5_cxn = np.load(os.path.join(data_path,'nmse_complex_COMPLEX_5.npy'))
            #     nmse_15_cxn = np.load(os.path.join(data_path,'nmse_complex_COMPLEX_15.npy'))
            #     nmse_35_cxn = np.load(os.path.join(data_path,'nmse_complex_COMPLEX_35.npy'))
            #     nmse_55_cxn = np.load(os.path.join(data_path,'nmse_complex_COMPLEX_55.npy'))
            #     plot2pgf([frequencies, nmse_5_cxn], 'nmse_5_cxn_COMPLEX', folder='results')
            #     plot2pgf([frequencies, nmse_15_cxn], 'nmse_15_cxn_COMPLEX', folder='results')
            #     plot2pgf([frequencies, nmse_35_cxn], 'nmse_35_cxn_COMPLEX', folder='results')
            #     plot2pgf([frequencies, nmse_55_cxn], 'nmse_55_cxn_COMPLEX', folder='results')

            #plot2pgf([frequencies, 10*np.log10(np.mean(nmse_5_lluis,axis=0))], 'nmse_pwd_freq_db_missing_'+str(n_missing), folder=pgf_dataset_path)

            print(f'Done {num_mics} mics! :)')

if __name__ == '__main__':
    main()



