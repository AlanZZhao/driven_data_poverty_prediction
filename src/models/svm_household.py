from src.data.interim_data import aX_h_train, bX_h_train, cX_h_train
from src.data.processed_data import ay_h_train, by_h_train, cy_h_train

from sklearn.externals import joblib
from pathlib import Path
import os
import pandas as pd
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(Path(__file__).parents[1]), 'models/')
INTERIM_DATA_DIR = os.path.join(os.path.dirname(Path(__file__).parents[1]), 'data/interim')

from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV

model_a = svm.SVC(probability=True)

model_a.fit(aX_h_train, ay_h_train)

# persist models
joblib.dump(model_a_cccv, os.path.join(MODEL_DIR, 'household_a_cccv_svm.pkl'))