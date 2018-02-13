from src.data.processed_data import a_i_test, a_h_test, b_i_test, b_h_test, c_i_test, c_h_test

from src.data.processed_data import aX_h_train, ay_h_train, bX_h_train, by_h_train, cX_h_train, cy_h_train
from src.data.processed_data import aX_i_train, ay_i_train, bX_i_train, by_i_train, cX_i_train, cy_i_train

from sklearn.externals import joblib
from pathlib import Path
import os
import pandas as pd
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(Path(__file__).parents[1]), 'models/')

# load models
model_a_cccv = joblib.load(os.path.join(MODEL_DIR, 'individual_a_cccv.pkl'))
model_b_cccv = joblib.load(os.path.join(MODEL_DIR, 'individual_b_cccv.pkl'))
model_c_cccv = joblib.load(os.path.join(MODEL_DIR, 'individual_c_cccv.pkl'))

# make predictions
a_preds = model_a_cccv.predict_proba(aX_i_train)
b_preds = model_b_cccv.predict_proba(bX_i_train)
c_preds = model_c_cccv.predict_proba(cX_i_train)

# add as features to dataset
for predictions, dataframe in zip([a_preds, b_preds, c_preds],[aX_i_train, bX_i_train, cX_i_train]):
    dataframe['poor_prediction'] = pd.Series(predictions[:, 1], index=dataframe.index)

def summarize_predictions(df):
    return df.groupby(['id'], as_index=False)['poor_prediction'].agg([np.min, np.mean, np.max])

# save as final household data
for
aX_h_train = aX_h_train.merge(individual_predictions, how='left', left_index=True, right_index=True)