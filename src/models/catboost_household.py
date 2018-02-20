from src.data.processed_data import a_i_test, a_h_test, b_i_test, b_h_test, c_i_test, c_h_test

from src.data.processed_data import aX_h_train, ay_h_train, bX_h_train, by_h_train, cX_h_train, cy_h_train
from src.data.processed_data import aX_i_train, ay_i_train, bX_i_train, by_i_train, cX_i_train, cy_i_train
from src.data.make_dataset import get_categorical_indices


import numpy as np
from catboost import CatBoostClassifier, cv, Pool
from sklearn.cross_validation import StratifiedKFold


a_indices, b_indices, c_indices = get_categorical_indices(aX_h_train), get_categorical_indices(bX_h_train), get_categorical_indices(cX_h_train)

model_a = CatBoostClassifier(nan_mode='Min')
model_a.fit(aX_h_train, ay_h_train, cat_features=a_indices)

model_b = CatBoostClassifier(nan_mode='Min')
model_b.fit(bX_h_train, by_h_train, cat_features=b_indices)

model_c = CatBoostClassifier(nan_mode='Min')
model_c.fit(cX_h_train, cy_h_train, cat_features=c_indices)


cv_data_a = cv(params = model_a.get_params(), pool = Pool(aX_h_train, ay_h_train, cat_features=a_indices))
a_score = cv_data_a['Logloss_test_avg'][-1]

cv_data_b = cv(params = model_b.get_params(), pool = Pool(bX_h_train, by_h_train, cat_features=b_indices))
b_score = cv_data_b['Logloss_test_avg'][-1]

cv_data_c = cv(params = model_c.get_params(), pool = Pool(cX_h_train, cy_h_train, cat_features=c_indices))
c_score = cv_data_c['Logloss_test_avg'][-1]