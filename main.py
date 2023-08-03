import os
import torch
import matplotlib.pyplot as plt
import glob
import numpy as np
#import params
import argparse
import pyroomacoustics as pra
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import datetime
# import network_lib_torch
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True
print(torch.__version__)
from dataset import SoundFieldDataset
import random

import utils
import sfun_torch as sfun


BASE_DIR ='/nas/home/fronchini/complex-sound-field'

#mask tutto di 1 e provo a ricostruire il sound field reconstruction


#TODO: does it make sense? 
def l1(y_true, y_pred):
        """Calculates the L1.

        Args:
            y_true: torch tensor
            y_pred: torch tensor

        Returns: torch tensor
        """
        #ipdb.set_trace()
        return torch.mean(torch.abs(y_pred - y_true)) 
        #TODO: is the same as axes in Keras, l1-loss di torch
        #TODO: considering only the magnitude and not the phase
        

def loss_valid(mask, y_true, y_pred):
    return l1(mask * y_true, mask * y_pred)

def loss_hole(mask, y_true, y_pred):
    return l1((1-mask) * y_true, (1-mask) * y_pred)
    

def soundfield_loss(mask, y_true, y_pred, valid_weight, hole_weight, device):
   
    y_true = y_true.to(device)
    
    valid_loss = loss_valid(mask, y_true, y_pred)
    hole_loss = loss_hole(mask, y_true, y_pred)

    loss_val = valid_weight*valid_loss + hole_weight*hole_loss

    return loss_val

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.json', help='JSON-formatted file with configuration parameters')
    parser.add_argument('--best_model_path', default='ComplexUNet', help='JSON-formatted file with configuration parameters')
    parser.add_argument('--mode', default='test', help='if testing or trainign')
    
    args = parser.parse_args()
    
    config_path = args.config
    best_model_path = args.best_model_path
    mode = args.mode
    
    # Load configuration
    if not os.path.exists(config_path):
        print('Error: No configuration file present at specified path.')
        return

    config = utils.load_config(config_path)
    print('Loaded configuration from: %s' % config_path)
    
    epochs = 3 if config["run"]["test"] else config["training"]["num_epochs"]
    lr = config["training"]["lr"]

    early_stop_patience = 100
    epoch_to_plot = 2  # Plot evey epoch_to_plot epochs
    #val_perc=0.2
    current_time = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Open SummaryWriter for Tensorboard
    writer = SummaryWriter(log_dir=(config["dataset"]["log_dir"] + config["training"]["session_id"]))
    dir_best_model = os.path.join(BASE_DIR, 'models', config["training"]["session_id"])
    os.makedirs(dir_best_model, exist_ok=True)
    saved_model_path = os.path.join(dir_best_model, best_model_path)
    

    # Imports to select GPU
    os.environ['CUDA_ALLOW_GROWTH'] = 'True'
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    cfg_dataset = config["dataset"]
    batch_size = 2 if config["run"]["test"] else config["training"]["batch_size"] 
    test_batch_size = 1
    
    # load dataset
    if mode == 'train':
        if config["run"]["test"] == 1:
            sf_train = SoundFieldDataset(dataset_folder=cfg_dataset["train_path"], xSample=cfg_dataset["xSamples"], ySample=cfg_dataset["ySamples"], factor=cfg_dataset["factor"])
            sf_val = SoundFieldDataset(dataset_folder=cfg_dataset["train_path"], xSample=cfg_dataset["xSamples"], ySample=cfg_dataset["ySamples"], factor=cfg_dataset["factor"])    
        else:
            dataset_filename_list = glob.glob(os.path.join(config["dataset"]["full_dataset_path"], "*.mat"))
            sf_train_list, sf_val_list  = train_test_split(dataset_filename_list, train_size=0.75, test_size=0.25, random_state=42)
        
            sf_val = SoundFieldDataset(set_file_list=sf_val_list, xSample=cfg_dataset["xSamples"], ySample=cfg_dataset["ySamples"], factor=cfg_dataset["factor"])
            sf_train = SoundFieldDataset(set_file_list=sf_train_list, xSample=cfg_dataset["xSamples"], ySample=cfg_dataset["ySamples"], factor=cfg_dataset["factor"])
        
        train_loader = torch.utils.data.DataLoader(sf_train,
                shuffle=True, 
                num_workers=4, 
                batch_size=batch_size) ##2 number of workers
    
        val_loader = torch.utils.data.DataLoader(sf_val, 
                shuffle=True, 
                num_workers=4,  
                batch_size=batch_size) ##2 number of workers  
        
    else:
        sf_test = SoundFieldDataset(dataset_folder=cfg_dataset["test_path"], xSample=cfg_dataset["xSamples"], ySample=cfg_dataset["ySamples"], factor=cfg_dataset["factor"])

        test_loader = torch.utils.data.DataLoader(sf_test, 
                shuffle=False, 
                num_workers=4,  ## ??
                batch_size=test_batch_size) ##2 number of workers  
     
    
    if mode == 'train': 
    
        # Load Model and hyperparams
        model = sfun.ComplexUnet(config["training"]) 
        model = model.to(device)
        
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Model parameters-->'+str(count_parameters(model)))
        
        count_parameters(model)
        
        # # Create optimizer + Scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, mode='min',threshold=0.00001)
        # need to consider this as well

        valid_weight = config["training"]["loss"]["valid_weight"]
        hole_weight = config["training"]["loss"]["hole_weight"]
        
        # # Training Loop
        def train_loop(train_loader, model, device):
            num_batches = len(train_loader.dataset)
            running_loss = 0
            model.train()
            
            for batch, (input_data, y_true) in enumerate(tqdm(train_loader)):
                
                input_data = input_data.to(device) 
                optimizer.zero_grad(set_to_none=True)

                y_pred = model(input_data)
                mask = input_data[:, 40:, :, :]
                
                loss = soundfield_loss(mask, y_true, y_pred, valid_weight, hole_weight, device)

                # Backpropagation
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            return running_loss/num_batches
        
        
        def plot_fig(input_data, y_pred, y_true, best_epoch):
            
            freq = 20
            
            input_data = input_data[0][freq]
            y_pred = y_pred[0][freq] 
            y_true = y_true[0][freq]
            
            
            fig = plt.figure(figsize=(30, 10))
            plt.subplot(131)
            plt.imshow(np.real((input_data.detach().cpu())), aspect='auto')
            plt.colorbar()
            plt.title(f'Masked Sound Field, epoch: {best_epoch}')
            plt.tight_layout()
            
            plt.subplot(132)
            plt.imshow(np.real((y_pred.detach().cpu())), aspect='auto')
            plt.colorbar()
            plt.title(f'Sound Field Reconstructed: {best_epoch}')
            plt.tight_layout()
            
            plt.subplot(133)
            plt.imshow(np.real((y_true.detach().cpu())), aspect='auto')
            plt.colorbar()
            plt.title(f'Sound Field Ground Truth: {best_epoch}')
            plt.tight_layout()
            
            writer.add_figure("Sound Field Generation", fig, best_epoch)
            

        # # Validation Loop
        val_loss_best = 0
        
        def val_loop(val_loader, model, epoch, device):
            num_batches = len(val_loader.dataset)
            running_loss = 0
            model.eval()
            
            input_data_to_plot = None
            y_true_to_plot = None
            
            #batch_to_plot = np.random.randint(0, num_batches)
            
            with torch.no_grad():
                
                for batch, (input_data, y_true) in enumerate(tqdm(val_loader)):
        
                    input_data = input_data.to(device)
                    y_pred = model(input_data)
                    mask = input_data[:, 40:, :, :]
                
                    # get only one random
                    loss = soundfield_loss(mask, y_true, y_pred, valid_weight, hole_weight, device)
                    running_loss += loss.item()
                    
                    
                    input_data_to_plot = input_data
                    y_true_to_plot = y_true
            
            #plot_fig(input_data, y_pred, y_true, epoch) # model best -> need to be added to the training loop, right?
            return running_loss/num_batches, input_data_to_plot, y_true_to_plot

        plot_val = True
        
        # Training
        for n_e in tqdm(range(epochs)):
            
            train_loss = train_loop(train_loader,model,device)
            val_loss, input_data_tp, y_true_tp = val_loop(val_loader, model, n_e, device)
            
            scheduler.step(val_loss)
            
            # Write to tensorboard
            writer.add_scalar('Loss/train',  train_loss, n_e)
            writer.add_scalar('Loss/val', val_loss, n_e)

            # Handle saving best model + early stopping
            if n_e == 0:
                val_loss_best = val_loss
                early_stop_counter = 0
                saved_model_path = saved_model_path + "_" + str(n_e)
                torch.save(model.state_dict(), saved_model_path)
            if n_e > 0 and val_loss < val_loss_best:
                saved_model_path = saved_model_path.split('_')[0] + "_" + str(n_e)
                torch.save(model.state_dict(), saved_model_path)
                val_loss_best = val_loss
                #print(f'Model saved epoch{n_e}')
                early_stop_counter = 0
            else:
                early_stop_counter +=1
                print('Patience status: ' + str(early_stop_counter) + '/' + str(early_stop_patience))
            
            # Early stopping
            if early_stop_counter > early_stop_patience:
                print('Training finished at epoch '+str(n_e))
                break

            # # At the end of every n_e epochs we print losses and compute soundfield plots and send them to tensorboard
            if not n_e % epoch_to_plot and plot_val and n_e > 0:
                print('Train loss: ' + str(train_loss))
                print('Val loss: ' + str(val_loss))

                # n_s = np.random.randint(0, src_val.shape[0])
                model_best = sfun.ComplexUnet(config["training"]).to(device)
                model_best.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', saved_model_path)))
                n_epoch_model_saved = saved_model_path.split('_')[1]
                print(f"Model loaded at epoch: {n_epoch_model_saved}")
                
                y_pred_tp = model_best(input_data_tp)
                plot_fig(input_data_tp, y_pred_tp, y_true_tp, n_epoch_model_saved)
    else: 
        # inference time loop
        evaluation_model = sfun.ComplexUnet(config["training"])
        
        evaluation_model.load_state_dict(torch.load("/nas/home/fronchini/complex-sound-field/models/2acti/ComplexUNet_570"))
        evaluation_model = evaluation_model.to(device)
        
        results_dic = {
            'nsme': []
        }
        
        frequencies = utils.get_frequencies()

        for batch, (input_data, y_true) in enumerate(tqdm(test_loader)):
            evaluation_model.eval()
            with torch.no_grad():
                input_data = input_data.to(device)
                y_true = y_true.to(device)
                y_pred = evaluation_model(input_data)
                mask = input_data[:, 40:, :, :]
                for freq in frequencies:
                    nmse = utils.NMSE_fun(y_pred[:, freq, :, :], y_true[:, freq, :, :])
                    results_dic['nmse'].append(nmse)
    
    # this nee to be done for every room and then averaged       
    # plt.figure()
    # plt.plot()
    # save plot
            

if __name__ == '__main__':
    main()



