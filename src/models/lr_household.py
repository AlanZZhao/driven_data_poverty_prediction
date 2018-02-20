from src.data.interim_data import aX_h_train_resampled, ay_h_train_resampled, bX_h_train_resampled, by_h_train_resampled, cX_h_train_resampled, cy_h_train_resampled
from src.data.interim_data import a_h_test, b_h_test, c_h_test
from src.data.interim_data import aX_h_train, bX_h_train, cX_h_train
from src.data.processed_data import ay_h_train, by_h_train, cy_h_train

from sklearn.externals import joblib
from pathlib import Path
import os
import pandas as pd
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(Path(__file__).parents[1]), 'models/')

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
model_a= lr.fit(aX_h_train, ay_h_train)
model_b= lr.fit(bX_h_train, by_h_train)
model_c= lr.fit(cX_h_train, cy_h_train)

joblib.dump(model_a, os.path.join(MODEL_DIR, 'household_a_lr_cont.pkl'))