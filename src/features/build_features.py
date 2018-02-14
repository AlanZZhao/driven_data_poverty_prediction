from src.data.processed_data import a_i_test, a_h_test, b_i_test, b_h_test, c_i_test, c_h_test

from src.data.processed_data import aX_h_train, ay_h_train, bX_h_train, by_h_train, cX_h_train, cy_h_train
from src.data.processed_data import aX_i_train, ay_i_train, bX_i_train, by_i_train, cX_i_train, cy_i_train

from sklearn.externals import joblib
from pathlib import Path
import os
import pandas as pd
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(Path(__file__).parents[1]), 'models/')
INTERIM_DATA_DIR = os.path.join(os.path.dirname(Path(__file__).parents[1]), 'data/interim')


# load models
model_a_cccv = joblib.load(os.path.join(MODEL_DIR, 'individual_a_cccv.pkl'))
model_b_cccv = joblib.load(os.path.join(MODEL_DIR, 'individual_b_cccv.pkl'))
model_c_cccv = joblib.load(os.path.join(MODEL_DIR, 'individual_c_cccv.pkl'))

# make predictions, add as to individual dataset
a_preds = model_a_cccv.predict_proba(aX_i_train)
b_preds = model_b_cccv.predict_proba(bX_i_train)
c_preds = model_c_cccv.predict_proba(cX_i_train)

for predictions, dataframe in zip([a_preds, b_preds, c_preds],[aX_i_train, bX_i_train, cX_i_train]):
    dataframe['poor_prediction'] = pd.Series(predictions[:, 1], index=dataframe.index)

# repeat for test data
a_preds = model_a_cccv.predict_proba(a_i_test)
b_preds = model_b_cccv.predict_proba(b_i_test)
c_preds = model_c_cccv.predict_proba(c_i_test)

for predictions, dataframe in zip([a_preds, b_preds, c_preds],[a_i_test, b_i_test, c_i_test]):
    dataframe['poor_prediction'] = pd.Series(predictions[:, 1], index=dataframe.index)

# extract summary statistics and add to household data

def summarize_predictions(df):
    df.reset_index(inplace=True)
    return df.groupby(['id'], as_index=False)['poor_prediction'].agg([np.min, np.mean, np.max])

aX_i_train, bX_i_train, cX_i_train = summarize_predictions(aX_i_train), summarize_predictions(bX_i_train), summarize_predictions(cX_i_train)

individual_predictions = pd.concat([aX_i_train, bX_i_train, cX_i_train])

a_i_test, b_i_test, c_i_test = summarize_predictions(a_i_test), summarize_predictions(b_i_test), summarize_predictions(c_i_test)

individual_predictions_test = pd.concat([a_i_test, b_i_test, c_i_test])

# save as final household data
aX_h_train = aX_h_train.merge(individual_predictions, how='left', left_index=True, right_index=True)
bX_h_train = bX_h_train.merge(individual_predictions, how='left', left_index=True, right_index=True)
cX_h_train = cX_h_train.merge(individual_predictions, how='left', left_index=True, right_index=True)

a_h_test = a_h_test.merge(individual_predictions_test, how='left', left_index=True, right_index=True)
b_h_test = b_h_test.merge(individual_predictions_test, how='left', left_index=True, right_index=True)
c_h_test = c_h_test.merge(individual_predictions_test, how='left', left_index=True, right_index=True)

aX_h_train.to_csv(os.path.join(INTERIM_DATA_DIR, 'aX_h_train.csv'))

bX_h_train.to_csv(os.path.join(INTERIM_DATA_DIR, 'bX_h_train.csv'))

cX_h_train.to_csv(os.path.join(INTERIM_DATA_DIR, 'cX_h_train.csv'))

a_h_test.to_csv(os.path.join(INTERIM_DATA_DIR, 'a_h_test.csv'))

b_h_test.to_csv(os.path.join(INTERIM_DATA_DIR, 'b_h_test.csv'))

c_h_test.to_csv(os.path.join(INTERIM_DATA_DIR, 'c_h_test.csv'))