#!/usr/bin/env python
# coding: utf-8

"""
Created on Monday June 2 2025

@author: Pratik

ConvLSTM Training Script for SST or Precipitation Data
-------------------------------------------------------

This script trains a convolutional LSTM (ConvLSTM) model on datasets in tensor format such as 
precipitation data example here.

Command-Line Arguments:
-----------------------
--train      : bool (default=False)   
    Specify "True" to train the model.

--full_data      : bool (default=False)   
    Specify "True" to train the model with full data.

--epochs      : int (default=20)   
    Number of training epochs to run.

--lr          : float (default=0.0025)   
    Learning rate for the optimizer.

--batch_size  : int (default=20)   
    Batch size for training.

--step_size   : int (default=3)   
    Step size for learning rate scheduler (after how many epochs to decay the LR).

--gamma       : float (default=0.5)   
    Multiplicative factor for learning rate decay in the scheduler.

--data        : str (default="sst")   
    Dataset name. Valid options are "sst" or "precip".
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
# torch.autograd.set_detect_anomaly(True)
from timeit import default_timer
import sys
import os
import argparse
# Add the utils directory to the path
from ConvLSTM import *

torch.manual_seed(0)
np.random.seed(0)

def main():
    parser = argparse.ArgumentParser(description='Train ConvLSTM on SST data')
    parser.add_argument('--train', action='store_true', help='Train the model (default: False if not specified)')
    parser.add_argument('--full_data', action='store_true', help='Train the model with full data (default: False if not specified)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0025, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--step_size', type=int, default=3, help='Step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma value.')

    args = parser.parse_args()
    print("Creating necessary folders ......... ")
        
    folder_path = "pred"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

    folder_path = "models"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")
        
    # Use the arguments
    epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    step_size = args.step_size
    gamma = args.gamma
    S = 64

    # print(epochs, learning_rate, step_size, gamma)

    T_in = 3
    T = 6
    train = args.train
    full_data = args.full_data
    ### location vectors 
    print(train)

    ################################################################
    # load data
    ################################################################
    print('loading dataset ...')
    if full_data:

        data = np.load("datasets/precipitation_simulated_data.npy")
        
        data_train, data_test = train_test_split(data, train_size=3900, random_state=42)
    else:
        data = np.load("datasets/precip-data-sample.npy")
        
        data_train, data_test = train_test_split(data, train_size=50, random_state=42)
    # ntrain = data_train.shape[0]
    ntest = data_test.shape[0]
    # train_a = torch.tensor(data_train[:,:,:,:T_in],dtype=torch.float)
    # train_a = train_a.repeat([1,1,1,T_in,1])
    train_u = torch.tensor(data_train[:,:,:,5],dtype=torch.float)
    test_u = torch.tensor(data_test[:,:,:,5],dtype=torch.float)

    x_train = torch.tensor(data_train[:,:,:,:T_in],dtype=torch.float).permute(0,3,1,2)
    x_test = torch.tensor(data_test[:,:,:,:T_in],dtype=torch.float).permute(0,3,1,2)
    print('Training data shape: {}'.format(x_train.shape))
    print('Test output shape: {}'.format(test_u.shape))
    

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, train_u), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, test_u), batch_size=batch_size, shuffle=False)

    print('data-preprocessing finished.')
    # device = torch.device('cuda')

    model = ConvLSTM2D_Model(T_in,1).cuda()
    
    ################################################################
    # training and evaluation
    ################################################################

    optimizer2 = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=step_size, gamma=gamma)

    best_val_loss = float('inf')
    interval = 5
    torch.manual_seed(329)
    if train:
        print("training started --- ConvLSTM training phase")
        for ep in range(epochs):
            
            model.train()
            # cov_model.train()
            t1 = default_timer()
            train_mse = 0
            train_l2 = 0
            for x, y in train_loader:
                x, y = x.cuda(), y.cuda()
                bs = x.shape[0]
                optimizer2.zero_grad()
                out = model(x)
                
                mse = F.mse_loss(out.view(-1), y.view(-1), reduction='mean')
                mse.backward()
                optimizer2.step()
                train_mse += mse.item()
                # train_l2 += loss.item()
            
            scheduler2.step()
            model.eval()
            # cov_model.eval()
            test_l2 = 0.0
            count = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.cuda(), y.cuda()

                    out = model(x)
                    test_l2 +=  F.mse_loss(out.view(-1), y.view(-1), reduction='mean').item()
                    count +=1
            train_mse /= len(train_loader)
            test_l2 /= count

            t2 = default_timer()
            print(f"Epoch: {ep} | Time: {t2 - t1:.2f}s | Train MSE: {train_mse:.4f} | Test Loss (L2): {test_l2:.4f}")
            # Early stopping
            if train_mse < best_val_loss:
                best_val_loss = train_mse
                epochs_without_improvement = 0
                # Save the model
                torch.save(model.state_dict(), 'models/convlstm_model.pth')
                # torch.save(cov_model.state_dict(), 'models/burger-cov-lcl_dat-nonlcl.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 6:
                    print(f"Early stopping at epoch {ep+1}")
                    break
        model.load_state_dict(torch.load('models/convlstm_model.pth', weights_only=True))
        print("Generating predictions ...")
        pred = torch.zeros_like(test_u)
        index = 0
        test_l2 = 0
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, test_u), batch_size=1, shuffle=False)
        with torch.no_grad():
            for x, y in test_loader:
                
                x, y = x.cuda(), y.cuda()

                out = model(x)
                pred[index] = out
                test_l2 += F.mse_loss(out.view(-1), y.view(-1), reduction='mean')
                index = index + 1
        print("average loss is : {}".format(test_l2/index))
        pred = pred.cpu().numpy()
        np.save("pred/ConvLSTM-pred.npy",pred)
        np.save("pred/ConvLSTM-test.npy", test_u)
    else:
        model_path = 'models/convlstm_model.pth'
        print("Generating predictions ...")
        if not os.path.exists(model_path):
            print(f"Model file not found at '{model_path}'.")
            print("Please train the model first by running the script with the '--train' flag set to True.")
            sys.exit(1)

        model.load_state_dict(torch.load(model_path, weights_only=True))
        
        pred = torch.zeros_like(test_u)
        index = 0
        test_l2 = 0
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_test, test_u),
            batch_size=1,
            shuffle=False
        )
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x)
                pred[index] = out
                test_l2 += F.mse_loss(out.view(-1), y.view(-1), reduction='mean')
                index += 1

        print("Average loss is: {:.6f}".format(test_l2 / index))
        pred = pred.cpu().numpy()
        np.save("pred/ConvLSTM-pred.npy", pred)
        np.save("pred/ConvLSTM-test.npy", test_u)


if __name__ == '__main__':
    main()