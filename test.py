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
import ipdb

import utils
import sfun_torch as sfun

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

    args = parser.parse_args()

    config_path = args.config
    # Load configuration
    if not os.path.exists(config_path):
        print('Error: No configuration file present at specified path.')
        return

    config = utils.load_config(config_path)
    print('Loaded configuration from: %s' % config_path)
    
    base_dir = config["dataset"]["base_dir"]
    


    # Imports to select GPU
    os.environ['CUDA_ALLOW_GROWTH'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = config["run"]["gpu"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    cfg_dataset = config["dataset"]
    test_batch_size = 1
    do_normalize = config["dataset"]["do_normalize"]

    # load dataset
    for num_mics in [5, 15, 35, 55]:

        print('Computing for '+str(num_mics)+' microphones')
        sf_test = SoundFieldDataset(dataset_folder=cfg_dataset["test_path"], xSample=cfg_dataset["xSamples"],
                                    ySample=cfg_dataset["ySamples"], factor=cfg_dataset["factor"],do_test=True, num_mics=num_mics, do_normalize=do_normalize)

        test_loader = torch.utils.data.DataLoader(sf_test,
                                                    shuffle=False,
                                                    num_workers=0,  # torch.cuda.device_count() *4,  ## ??
                                                    batch_size=test_batch_size,
                                                    pin_memory=True)


        # inference time loop
        model = sfun.ComplexUnet(config["training"])
        model_name = config["training"]["session_id"]
        
        evaluation_folder_path = os.path.join(base_dir, 'models', str(model_name))
        
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
                    
                    plot_file_path = os.path.join(results_eval_path, f'{num_mics}_1.png')
                    plt.savefig(plot_file_path) 


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
                    
                    plot_file_path = os.path.join(results_eval_path, f'{num_mics}_2.png')
                    plt.savefig(plot_file_path) 
            idx = idx +1
            if idx == 50:
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
       
        plot_file_path = os.path.join(results_eval_path, 'NMSE_[dB].png')
        plt.savefig(plot_file_path) 
        
        plt.show()



if __name__ == '__main__':
    main()



