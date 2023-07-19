import os
os.environ['CUDA_VISIBLE_DEVICES'] ='3'
import torch
import matplotlib.pyplot as plt
import numpy as np
#import params
import argparse
import pyroomacoustics as pra
#from sklearn.model_selection import train_test_split
from tqdm import tqdm
import datetime
# import network_lib_torch
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True
print(torch.__version__)
from dataset import SoundFieldDataset
import ipdb

import utils
import sfun_torch as sfun


BASE_DIR ='/nas/home/fronchini/complex-sound-field'


#TODO: does it make sense? 
def l1(y_true, y_pred):
        """Calculates the L1.

        Args:
            y_true: torch tensor
            y_pred: torch tensor

        Returns: torch tensor
        """
        #ipdb.set_trace()
        return torch.mean(torch.abs(y_pred - y_true), dim=[1,2,3]) 
        #TODO: is the same as axes in Keras, l1-loss di torch
        #TODO: considering only the magnitude and not the phase
        


def loss_valid(mask, y_true, y_pred):
    return l1(mask * y_true, mask * y_pred)

def loss_hole(mask, y_true, y_pred):
    return l1((1-mask) * y_true, (1-mask) * y_pred)
    

def soundfield_loss(mask, y_true, y_pred, valid_weight, hole_weight, device):
    
    #ipdb.set_trace()
    y_true = y_true.to(device)
    
    valid_loss = loss_valid(mask, y_true, y_pred)
    hole_loss = loss_hole(mask, y_true, y_pred)

    loss_val = valid_weight*valid_loss + hole_weight*hole_loss
    

    return loss_val

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.json', help='JSON-formatted file with configuration parameters')
    args = parser.parse_args()
    
    config_path = args.config
    
    # Load configuration
    if not os.path.exists(config_path):
        print('Error: No configuration file present at specified path.')
        return

    config = utils.load_config(config_path)
    print('Loaded configuration from: %s' % config_path)
    
    epochs = config["training"]["num_epochs"]
    lr = config["training"]["lr"]

    early_stop_patience = 10
    epoch_to_plot = 1  # Plot evey epoch_to_plot epochs
    #val_perc=0.2
    current_time = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Open SummaryWriter for Tensorboard
    writer = SummaryWriter(log_dir=config["dataset"]["log_dir"])
    saved_model_path = os.path.join(BASE_DIR, 'models', "ComplexUNet")


    # Imports to select GPU
    os.environ['CUDA_ALLOW_GROWTH'] = 'True'
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    
    # load dataset
    cfg_dataset = config["dataset"]
    sf_train = SoundFieldDataset(cfg_dataset["train_path"], cfg_dataset["xSamples"], cfg_dataset["ySamples"], cfg_dataset["factor"])
    sf_val = SoundFieldDataset(cfg_dataset["train_path"], cfg_dataset["xSamples"], cfg_dataset["ySamples"], cfg_dataset["factor"])
    
    train_loader = torch.utils.data.DataLoader(sf_train, shuffle=False, num_workers=0, batch_size=1) ##2 number of workers
    val_loader = torch.utils.data.DataLoader(sf_val, shuffle=False, num_workers=0, batch_size=1) ##2 number of workers

    # Load Model and hyperparams
    model = sfun.ComplexUnet(config["training"])
    model = model.to(device)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model parameters-->'+str(count_parameters(model)))
    
    count_parameters(model)
    
    # # Create optimizer + Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, mode='min',threshold=0.00001)
    # need to consider this as well


    # # Create Dataloaders
    batch_size = int(config["training"]["batch_size"]) * torch.cuda.device_count()
    print(f"batch size is: {batch_size}")

    
    valid_weight = config["training"]["loss"]["valid_weight"]
    hole_weight = config["training"]["loss"]["hole_weight"]
    
    # # Training Loop
    def train_loop(train_loader, model, device):
        num_batches = len(train_loader.dataset)
        running_loss = 0
        model.train()
        
        for batch, (input_data, y_true) in enumerate(tqdm(train_loader)):
            
            ipdb.set_trace()
            
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

    # # Validation Loop
    val_loss_best = 0
    def val_loop(val_loader, model, device):
        num_batches = len(val_loader.dataset)
        running_loss = 0
        model.eval()
        with torch.no_grad():
            
            for batch, (input_data, y_true) in enumerate(tqdm(val_loader)):
                input_data = input_data.to(device)
                y_pred = model(input_data)
                mask = input_data[:, 40:, :, :]
            
                loss = soundfield_loss(mask, y_true, y_pred, valid_weight, hole_weight, device)
                running_loss += loss.item()
        
        return running_loss/num_batches


    # print('Start training')
    plot_val = True
    for n_e in tqdm(range(epochs)):
        train_loss = train_loop(train_loader,model,device)
        val_loss = val_loop(val_loader, model, device)
        scheduler.step(val_loss)
        
        # Write to tensorboard
        writer.add_scalar('Loss/train',  train_loss, n_e)
        writer.add_scalar('Loss/val', val_loss, n_e)

        # Handle saving best model + early stopping
        # if n_e == 0:
        #     val_loss_best = val_loss
        #     early_stop_counter = 0
        # if n_e > 0 and val_loss < val_loss_best:
        #     torch.save(model.state_dict(), saved_model_path)
        #     val_loss_best = val_loss
        #     print(f'Model saved epoch{n_e}')
        #     early_stop_counter = 0
        # else:
        #     early_stop_counter +=1
        #     print('Patience status: ' + str(early_stop_counter) + '/' + str(early_stop_patience))
        # # Early stopping
        # if early_stop_counter > early_stop_patience:
        #     print('Training finished at epoch '+str(n_e))
        #     break

        # # At the end of every n_e epochs we print losses and compute soundfield plots and send them to tensorboard
        # if not n_e % epoch_to_plot and plot_val and n_e > 0:
        #     print('Train loss: ' + str(train_loss))
        #     print('Val loss: ' + str(val_loss))

        #     n_s = np.random.randint(0, src_val.shape[0])
        #     model_best = network_lib_torch.ComplexNetRes(input_size=(1, N_cp, 64), output_size=(1, G_B.shape[1], 64)).to(device)
        #     model_best.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', args.model_name)))

        #     d_hat = model_best(torch.from_numpy(P_val[n_s]).unsqueeze(0).unsqueeze(1).to(device))
        #     #idx_f = np.random.randint(0, params.M)
        #     idx_f = 30  # FIXEDD to 1000 Hzzz

        #     P_hat_B = torch.einsum('bij, kij-> bkj', d_hat, torch.from_numpy(G_B_eval).to(device))[0]
        #     P_hat_D = torch.einsum('bij, kij-> bkj', d_hat, torch.from_numpy(G_D_eval).to(device))[0]

        #     # Compute GT Bright
        #     P_gt_B = np.zeros((params.M, params.F), dtype=np.complex64)
        #     xs = src_val[n_s] + params.room_shift
        #     room, mic_locs_pra = pra.AnechoicRoom(fs=params.f_s), params.r_m_b_shift.T
        #     room.add_microphone(mic_locs_pra)
        #     room.add_source(xs)
        #     room.compute_rir()
        #     for m in range(params.M):
        #         sig_rir = room.rir[m][0][params.pyroom_buffer:]
        #         P_gt_B[m] = np.fft.rfft(sig_rir, n=params.nfft)[params.idx_freq]

        #     # GT dark
        #     P_gt_D = np.zeros_like(P_gt_B, dtype=np.complex64)

        #     fig = plt.figure(figsize=(10, 10))
        #     plt.subplot(221)
        #     plt.imshow(np.imag(np.reshape(P_gt_B[:, idx_f], (8, 8))),
        #                aspect='auto'), plt.colorbar(), plt.tight_layout()
        #     plt.xlabel('$x [m]$'), plt.ylabel('$y [m]$'), plt.title('$\mathrm{P_B}$')
        #     plt.subplot(222)
        #     plt.imshow(np.imag(np.reshape(P_gt_D[:, idx_f], (8, 8))),
        #                aspect='auto'), plt.colorbar(), plt.tight_layout()
        #     plt.xlabel('$x [m]$'), plt.ylabel('$y [m]$'), plt.title('$\mathrm{P_D}$')
        #     plt.subplot(223)
        #     plt.imshow(np.imag(np.reshape(P_hat_B[:, idx_f].detach().cpu(), (8, 8))),
        #                aspect='auto'), plt.colorbar(), plt.tight_layout()
        #     plt.xlabel('$x [m]$'), plt.ylabel('$y [m]$'), plt.title('$\mathrm{\hat{P}_B}$')
        #     plt.subplot(224)
        #     plt.imshow(np.imag(np.reshape(P_hat_D[:, idx_f].detach().cpu(), (8, 8))),
        #                aspect='auto'), plt.colorbar(), plt.tight_layout()
        #     plt.xlabel('$x [m]$'), plt.ylabel('$y [m]$'), plt.title('$\mathrm{\hat{P}_D}$')
        #     # Send image to tensorboard
        #     writer.add_figure('Soundfield Plot',fig,n_e)

        #     # Send image to tensorboard
        #     fig_filters = plt.figure()
        #     plt.plot(d_hat[0, :, :].detach().cpu())
        #     writer.add_figure('Filters Plot',fig_filters,n_e)


if __name__ == '__main__':
    main()



