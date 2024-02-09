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
import glob

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
    do_plot = False
   

    # Imports to select GPU
    os.environ['CUDA_ALLOW_GROWTH'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = config["run"]["gpu"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    cfg_dataset = config["dataset"]
    test_batch_size = 1
    do_normalize = config["dataset"]["do_normalize"]

    # Use glob to get a list of sub-folders
    
    # "test_path": "/nas/home/fronchini/sound-field-neural-network/datasets/real_soundfields",
    
    #test_set_list = glob.glob(os.path.join(cfg_dataset["test_path"], '*'))
    
    # Filter out non-directory entries
    #test_set_list = [folder for folder in test_set_list if os.path.isdir(folder)]
    
    
    #num_mics_list = cfg_dataset["num_mics_list"][2:3]   
    num_mics_list = [5]
    print(f"Considering the following number of microphones: {num_mics_list}")
    
    ipdb.set_trace()
    for num_mics in num_mics_list:
        print(f'Computing for: {num_mics} microphones')
        
        # in case of multiple T60 in simulated data:
        #for current_test_path in test_set_list:
        for T60 in [0.4]:
            #print(current_test_path)
            #T60 = (current_test_path.split("/")[-1]).split("_")[-1]
            test_path = os.path.join(cfg_dataset['test_path'], f'T60_{T60}')
            

            #print(f'Computing for T60 = {T60}')
            sf_test = SoundFieldDataset(dataset_folder=test_path, xSample=cfg_dataset["xSamples"],
                                        ySample=cfg_dataset["ySamples"], factor=cfg_dataset["factor"],do_test=True, num_mics=num_mics, do_normalize=do_normalize)

            test_loader = torch.utils.data.DataLoader(sf_test,
                                                        shuffle=False,
                                                        num_workers=0,  # torch.cuda.device_count() *4,  ## ??
                                                        batch_size=test_batch_size,
                                                        pin_memory=True)

            # inference time loop
            model = sfun.ComplexUnet(config["training"])
            model_name = config["training"]["session_id"]
            
            # evaluation folder path
            evaluation_folder_path = os.path.join(base_dir, 'models', str(model_name))
            
            # model used for evaluating
            evaluation_model_path = os.path.join(evaluation_folder_path, 'ComplexUNet')
            
            # results path
            results_eval_path = os.path.join(evaluation_folder_path, 'results', 'T60', f'n_mics_{num_mics}')
            os.makedirs(results_eval_path, exist_ok=True)
            
            model.load_state_dict(torch.load(evaluation_model_path, map_location='cuda:0'))
            model = model.to(device)

            results_dic = {
                'nmse': [],
                'nmse_complex': [],
                'ssim': []
            }
            
            frequencies = utils.get_frequencies()
            #idx = 0
            
            for _, (input_data, y_true) in enumerate(tqdm(test_loader)):
                model.eval()
                with torch.no_grad():
                    input_data = input_data.to(device)
                    y_true = y_true.to(device)
                    y_pred = model(input_data)
                    #mask = input_data[:, 40:, :, :]

                    # compute the metrics nmse, nmse_complex and SSIM
                    nmse = utils.NMSE_fun(y_pred[0].cpu().numpy(), y_true[0].cpu().numpy())
                    nmse_complex = utils.NMSE_complex_fun(y_pred[0].cpu().numpy(), y_true[0].cpu().numpy())
                    ssim_metric = utils.SSIM_fun(y_pred[0].cpu().numpy(), y_true[0].cpu().numpy())

                    results_dic['nmse'].append(nmse)
                    results_dic['nmse_complex'].append(nmse_complex)
                    results_dic['ssim'].append(ssim_metric)

                
            average_nsme = 10 * np.log10(np.mean(results_dic['nmse'], axis=0))
            average_nsme_complex = 10 * np.log10(np.mean(results_dic['nmse_complex'], axis=0))
            
            # calcolo della standard deviation 
            std_abs = 10 * np.log10(np.std(results_dic['nmse'], axis=0))
            std_complex = 10 * np.log10(np.std(results_dic['nmse_complex'], axis=0))

            # save nsme
            filename_path = os.path.join(results_eval_path, f'nmse_complex_{T60}.npy')
            np.save(filename_path, average_nsme, allow_pickle=False)

            # save complex nmse (ugly names)
            filename_path = os.path.join(results_eval_path, f'nmse_complex_COMPLEX_{T60}.npy')
            np.save(filename_path, average_nsme_complex, allow_pickle=False)
            
            # save the std for nmse 
            filename_path = os.path.join(results_eval_path, f'std_complex_{T60}.npy')
            np.save(filename_path, std_abs, allow_pickle=False)
            
            # # save the std for nmse for complex values
            filename_path = os.path.join(results_eval_path, f'std_complex_COMPLEX_{T60}.npy')
            np.save(filename_path, std_complex, allow_pickle=False)
            
            
            average_ssim = np.mean(results_dic['ssim'], axis=0)
            # save ssim
            filename_path = os.path.join(results_eval_path, f'ssim_complex_{T60}.npy')
            np.save(filename_path, average_ssim, allow_pickle=False)

            x_values = frequencies

            # calculate the NMSE
            plt.figure(figsize=(14, 10))
            plt.plot(x_values, average_nsme)
            plt.xscale('log')


            plt.xlabel('$f [Hz]$'), plt.ylabel('$NMSE [dB]$')#, #plt.title('$\text{NMSE estimated from simulated data}$')
            plt.grid(which='both', linestyle='-', linewidth=0.5, color='gray')
        
            plot_file_path = os.path.join(results_eval_path, f'NMSE_[dB]_{T60}.png')
            plt.savefig(plot_file_path) 
            
            plt.show()



if __name__ == '__main__':
    main()



