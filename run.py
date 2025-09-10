import contextlib
import csv
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.data.preprocessing import preprocessing_location_polar_output
from src.model.dnn import FCDNN
from src.utils.losses import circular_mae, rmse
from src.utils.lr_scheduler import LRScheduler


# list of files 
data_dir = 'data/2nulls_05_6_correlation'
bf_files = []
null_files = []
w_files = []
files = os.listdir(data_dir)

for file in files:
    if file.endswith('.mat'):
        data_path = os.path.join(data_dir, file)
        if file.startswith('LCMV_BF'):
            bf_files.append(data_path)
        elif file.startswith('LCMV_null'):
            null_files.append(data_path)
        elif file.startswith('LCMV_weights'):
            w_files.append(data_path)
                
bf_files.sort()
null_files.sort()
w_files.sort()

# Training models for each dataset
# make the directory to save the artifacts
model_save_dir = os.path.join("E:/DL-based Near Field NCB/General Near Field Reorganized/Codebook/artifacts/models/2nulls_correlation", datetime.now().strftime('%Y_%m_%d_%H_%M'))
plot_save_dir = os.path.join("E:/DL-based Near Field NCB/General Near Field Reorganized/Codebook/artifacts/plots/2nulls_correlation", datetime.now().strftime('%Y_%m_%d_%H_%M'))
log_save_dir = os.path.join("E:/DL-based Near Field NCB/General Near Field Reorganized/Codebook/artifacts/logs/2nulls_correlation", datetime.now().strftime('%Y_%m_%d_%H_%M'))

os.mkdir(model_save_dir)
os.mkdir(plot_save_dir)
os.mkdir(log_save_dir)

for i in range(len(bf_files)):
    # load data
    bf_points = scipy.io.loadmat(bf_files[i])['bf_points']
    null_points = scipy.io.loadmat(null_files[i])['null_points']
    weights = scipy.io.loadmat(w_files[i])['W']
    
    # Find nan values
    nan_indices = np.unique(np.argwhere(np.isnan(weights))[:, 0])
    large_pow_indices= np.where(np.sum(np.abs(weights)**2, axis=1) > 10)[0]
    corrupted_indices = np.unique(np.concatenate((nan_indices, large_pow_indices)))
    
    # Remove corrupted data points
    bf_points = np.delete(bf_points, corrupted_indices, axis=0)
    null_points = np.delete(null_points, corrupted_indices, axis=0)
    weights = np.delete(weights, corrupted_indices, axis=0)
    
    # preprocessing 
    data, w_mag, w_phase = preprocessing_location_polar_output(bf_points, null_points, weights)
    train_data, test_data, train_w_mag, test_w_mag, train_w_phase, test_w_phase = train_test_split(data, w_mag, w_phase, test_size=0.2, random_state=42)
    del data, w_mag, w_phase
    
    # Training Phase model
    # Initialize the phase estimation model
    phase_model_architecture = [1024, 512, 512, 256, 128, 64]
    dropout = 0.1
    initial_lr = 0.05

    phase_model = FCDNN(
        num_layers=len(phase_model_architecture),
        units=phase_model_architecture,
        input_shape=train_data.shape[1],
        output_dim=24,
        dropout=dropout,
        loss=circular_mae,
        model_save_dir=f"{model_save_dir}/Codebook_Dataset_DNN_{phase_model_architecture}_Dropout_{dropout}_lr_{initial_lr}_Phase_Correlation_range_40_region_{i+1}")
    
    lr_scheduler = LRScheduler(list(range(200)), initial_lr, 1e-6, mode="exp", lr_step=0.97)

    with open(f"{log_save_dir}/Codebook_Dataset_DNN_{phase_model_architecture}_Dropout_{dropout}_lr_{initial_lr}_Phase_Correlation_range_40_region_{i+1}.txt", 'w') as f:
        with contextlib.redirect_stdout(f):
            phase_model.summary()
            loss_dict = phase_model.train(
                train_data, train_w_phase, 
                test_data, test_w_phase,
                epochs=200,
                batch_size=1000,
                lr=initial_lr,
                lr_scheduler=lr_scheduler,
                device="/GPU:0"
                )

    # Save the loss values
    loss_df = pd.DataFrame(loss_dict)
    loss_df.to_csv(f"{log_save_dir}/Loss_Codebook_Dataset_DNN_{phase_model_architecture}_Dropout_{dropout}_lr_{initial_lr}_Phase_Correlation_range_40_region_{i+1}.csv", index=False)
    
    # Plot the loss values
    fig = plt.figure(dpi=300, figsize=(5, 5))
    plt.yscale("log")
    plt.plot(loss_dict["train_infer"], "b", label="Train Loss")
    plt.plot(loss_dict["val"], "r", label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Losses for Phase Estimation')
    plt.grid("minor")
    plt.legend()
    plt.savefig(f"{log_save_dir}/Loss_Codebook_Dataset_DNN_{phase_model_architecture}_Dropout_{dropout}_lr_{initial_lr}_Phase_Correlation_range_40_region_{i+1}.png")
    
    
    # Training Magnitude model
    # Initialize the magnitude estimation model
    mag_model_architecture = [1024, 512, 512, 256, 128, 64]
    dropout = 0.1
    initial_lr = 0.05
    mag_model = FCDNN(
        num_layers=len(mag_model_architecture),
        units=mag_model_architecture,
        input_shape=train_data.shape[1],
        output_dim=24,
        dropout=dropout,
        loss=rmse,
        model_save_dir=f"{model_save_dir}/Codebook_Dataset_DNN_{mag_model_architecture}_Dropout_{dropout}_lr_{initial_lr}_Magnitude_Correlation_range_40_region_{i+1}")
    
    lr_scheduler = LRScheduler(list(range(200)), initial_lr, 1e-6, mode="exp", lr_step=0.97)
    
    with open(f"{log_save_dir}/Codebook_Dataset_DNN_{mag_model_architecture}_Dropout_{dropout}_lr_{initial_lr}_Magnitude_Correlation_range_40_region_{i+1}.txt", 'w') as f:
        with contextlib.redirect_stdout(f):
            mag_model.summary()
            loss_dict = mag_model.train(
                train_data, train_w_mag, 
                test_data, test_w_mag,
                epochs=200,
                batch_size=1000,
                lr=initial_lr,
                lr_scheduler=lr_scheduler,
                device="/GPU:0"
                )

    # Save the loss values
    loss_df = pd.DataFrame(loss_dict)
    loss_df.to_csv(f"{log_save_dir}/Loss_Codebook_Dataset_DNN_{mag_model_architecture}_Dropout_{dropout}_lr_{initial_lr}_Magnitude_Correlation_range_40_region_{i+1}.csv", index=False)
    
    # Plot the loss values
    fig = plt.figure(dpi=300, figsize=(5, 5))
    plt.yscale("log")
    plt.plot(loss_dict["train_infer"], "b", label="Train Loss")
    plt.plot(loss_dict["val"], "r", label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Losses for Magnitude Estimation')
    plt.grid("minor")
    plt.legend()
    plt.savefig(f"{log_save_dir}/Loss_Codebook_Dataset_DNN_{mag_model_architecture}_Dropout_{dropout}_lr_{initial_lr}_Magnitude_Correlation_range_40_region_{i+1}.png")
    