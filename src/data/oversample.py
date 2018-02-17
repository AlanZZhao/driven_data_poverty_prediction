from src.data.interim_data import aX_h_train, bX_h_train, cX_h_train
from src.data.processed_data import ay_h_train, by_h_train, cy_h_train

import os
from pathlib import Path
import numpy as np
from imblearn.over_sampling import RandomOverSampler

INTERIM_DATA_DIR = os.path.join(os.path.dirname(Path(__file__).parents[1]), 'data/interim')

# ROS returns NP arrays, not data frames
ros1 = RandomOverSampler()

aX_h_train_resampled, ay_h_train_resampled = ros1.fit_sample(aX_h_train, ay_h_train)

bX_h_train_resampled, by_h_train_resampled = ros1.fit_sample(bX_h_train, by_h_train)

cX_h_train_resampled, cy_h_train_resampled = ros1.fit_sample(cX_h_train, cy_h_train)

np.save(os.path.join(INTERIM_DATA_DIR, 'ay_h_train_resampled.npy'), ay_h_train_resampled)

np.save(os.path.join(INTERIM_DATA_DIR, 'by_h_train_resampled.npy'), by_h_train_resampled)

np.save(os.path.join(INTERIM_DATA_DIR, 'cy_h_train_resampled.npy'), cy_h_train_resampled)

np.save(os.path.join(INTERIM_DATA_DIR, 'aX_h_train_resampled.npy'), aX_h_train_resampled)

np.save(os.path.join(INTERIM_DATA_DIR, 'bX_h_train_resampled.npy'), bX_h_train_resampled)

np.save(os.path.join(INTERIM_DATA_DIR, 'cX_h_train_resampled.npy'), cX_h_train_resampled)