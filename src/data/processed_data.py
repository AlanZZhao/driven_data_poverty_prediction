import numpy as np
import pandas as pd
import os
from pathlib import Path

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(Path(__file__).parents[1]), 'data/processed')

aX_h_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'aX_h_train.csv'), index_col=['id'])
ay_h_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'ay_h_train.npy'))

bX_h_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'bX_h_train.csv'), index_col=['id'])
by_h_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'by_h_train.npy'))

cX_h_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'cX_h_train.csv'), index_col=['id'])
cy_h_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'cy_h_train.npy'))

aX_i_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'aX_i_train.csv'), index_col=['id', 'iid'])
ay_i_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'ay_i_train.npy'))

bX_i_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'bX_i_train.csv'), index_col=['id', 'iid'])
by_i_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'by_i_train.npy'))

cX_i_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'cX_i_train.csv'), index_col=['id', 'iid'])
cy_i_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'cy_i_train.npy'))

a_h_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'a_h_test.csv'), index_col=['id'])
b_h_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'b_h_test.csv'), index_col=['id'])
c_h_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'c_h_test.csv'), index_col=['id'])

a_i_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'a_i_test.csv'), index_col=['id', 'iid'])
b_i_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'b_i_test.csv'), index_col=['id', 'iid'])
c_i_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'c_i_test.csv'), index_col=['id', 'iid'])