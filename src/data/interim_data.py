import numpy as np
import pandas as pd
import os
from pathlib import Path

INTERIM_DATA_DIR = os.path.join(os.path.dirname(Path(__file__).parents[1]), 'data/interim')

aX_h_train = pd.read_csv(os.path.join(INTERIM_DATA_DIR, 'aX_h_train.csv'), index_col=['id'])

bX_h_train = pd.read_csv(os.path.join(INTERIM_DATA_DIR, 'bX_h_train.csv'), index_col=['id'])

cX_h_train = pd.read_csv(os.path.join(INTERIM_DATA_DIR, 'cX_h_train.csv'), index_col=['id'])

a_h_test = pd.read_csv(os.path.join(INTERIM_DATA_DIR, 'a_h_test.csv'), index_col=['id'])
b_h_test = pd.read_csv(os.path.join(INTERIM_DATA_DIR, 'b_h_test.csv'), index_col=['id'])
c_h_test = pd.read_csv(os.path.join(INTERIM_DATA_DIR, 'c_h_test.csv'), index_col=['id'])

ay_h_train_resampled = np.load(os.path.join(INTERIM_DATA_DIR, 'ay_h_train_resampled.npy'))
by_h_train_resampled = np.load(os.path.join(INTERIM_DATA_DIR, 'by_h_train_resampled.npy'))
cy_h_train_resampled = np.load(os.path.join(INTERIM_DATA_DIR, 'cy_h_train_resampled.npy'))

aX_h_train_resampled = np.load(os.path.join(INTERIM_DATA_DIR, 'aX_h_train_resampled.npy'))
bX_h_train_resampled = np.load(os.path.join(INTERIM_DATA_DIR, 'bX_h_train_resampled.npy'))
cX_h_train_resampled = np.load(os.path.join(INTERIM_DATA_DIR, 'cX_h_train_resampled.npy'))