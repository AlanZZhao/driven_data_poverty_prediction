# import test data
from src.data.interim_data import a_h_test, b_h_test, c_h_test
from src.data.submission import make_country_sub

from sklearn.externals import joblib
from pathlib import Path
import os
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(Path(__file__).parents[1]), 'models/')
SUBMISSION_DATA_DIR = os.path.join(os.path.dirname(Path(__file__).parents[1]), 'data/submission')

# load models
model_a = joblib.load(os.path.join(MODEL_DIR, 'household_a_lr_cont.pkl'))
model_b = joblib.load(os.path.join(MODEL_DIR, 'household_b_rf_oversample.pkl'))
model_c = joblib.load(os.path.join(MODEL_DIR, 'household_c_rf_oversample.pkl'))

# predict
preds_a = model_a.predict_proba(a_h_test)
preds_b = model_b.predict_proba(b_h_test)
preds_c = model_c.predict_proba(c_h_test)

# make submission
sub_a = make_country_sub(preds_a, a_h_test, 'A')
sub_b = make_country_sub(preds_b, b_h_test, 'B')
sub_c = make_country_sub(preds_c, c_h_test, 'C')

submission = pd.concat([sub_a, sub_b, sub_c]).reset_index()
submission.to_csv(os.path.join(SUBMISSION_DATA_DIR, 'full_model_rf_bc_lr_a_cont.csv'), index=False)
