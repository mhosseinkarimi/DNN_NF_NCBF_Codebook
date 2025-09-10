import os

import numpy as np
import scipy

from src.data.preprocessing import preprocessing_location_polar_output
from src.model.dnn import FCDNN
from src.utils.losses import circular_mae, rmse
from src.utils.misc import db2mag

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

region_index = 6
sample_idx = np.random.randint(0, 100000, size=10)

bf_points = scipy.io.loadmat(bf_files[region_index])['bf_points'][sample_idx]
null_points = scipy.io.loadmat(null_files[region_index])['null_points'][sample_idx]
weights = scipy.io.loadmat(w_files[region_index])['W'][sample_idx]

# Preprocessing
data, w_mag, w_phase = preprocessing_location_polar_output(bf_points, null_points, weights)

# print(os.listdir('E:/DL-based Near Field NCB/General Near Field Reorganized/Codebook/artifacts/models/2nulls_correlation/2025_05_08_15_56'))

# Loading Phase model
phase_model_architecture = [1024, 512, 512, 256, 128, 64]
phase_model = FCDNN(
    num_layers=len(phase_model_architecture),
    units=phase_model_architecture,
    input_shape=data.shape[1],
    output_dim=24,
    dropout=0.1,
    loss=circular_mae,
    )                  

phase_model.model.load_weights(
    f'E:/DL-based Near Field NCB/General Near Field Reorganized/Codebook/artifacts/models/2nulls_correlation/2025_05_08_15_56/Codebook_Dataset_DNN_[1024, 512, 512, 256, 128, 64]_Dropout_01_lr_005_Phase_Correlation_range_40_region_{region_index+1}'
)
phase_model.summary()

# Load Magnitude model
mag_model_architecture = [1024, 512, 512, 256, 128, 64]
mag_model = FCDNN(
    num_layers=len(mag_model_architecture),
    units=mag_model_architecture,
    input_shape=data.shape[1],
    output_dim=24,
    dropout=0.1,
    loss=rmse,
    )
mag_model.model.load_weights(
    f'E:/DL-based Near Field NCB/General Near Field Reorganized/Codebook/artifacts/models/2nulls_correlation/2025_05_16_11_37/Codebook_Dataset_DNN_[1024, 512, 512, 256, 128, 64]_Dropout_01_lr_005_Magnitude_Correlation_range_40_region_{region_index+1}'
)
mag_model.summary()

# Prediction
w_phase_pred = phase_model.model(data).numpy()
w_mag_pred = mag_model.model(data).numpy()
w_pred = db2mag(w_mag_pred) * np.exp(1j*w_phase_pred)

print(circular_mae(w_phase_pred, w_phase))
print(rmse(w_mag_pred, w_mag))

# sort the indices based on the cumulative error
cumulative_error = circular_mae(w_phase_pred, w_phase) + rmse(w_mag_pred, w_mag)/5
sorted_indices = np.argsort(cumulative_error)
# Select the top 3 indices with the lowest cumulative error
top_indices = sorted_indices[:3]

print(f'Top 3 indices with lowest cumulative error: {top_indices}')
print(f'Corresponding phase errors: {circular_mae(w_phase_pred[top_indices], w_phase[top_indices])}')
print(f'Corresponding magnitude errors: {rmse(w_mag_pred[top_indices], w_mag[top_indices])}')

# Save test cases
scipy.io.savemat(f'artifacts/test cases/2null_correlation/bf_points_region_{region_index+1}_top3.mat', {'bf_points': bf_points[top_indices]})
scipy.io.savemat(f'artifacts/test cases/2null_correlation/null_points_region_{region_index+1}_top3.mat', {'null_points': null_points[top_indices]})
scipy.io.savemat(f'artifacts/test cases/2null_correlation/w_pred_region_{region_index+1}_top3.mat', {'w_pred': w_pred[top_indices]})
scipy.io.savemat(f'artifacts/test cases/2null_correlation/w_lcmv_region_{region_index+1}_top3.mat', {'w': weights[top_indices]})
