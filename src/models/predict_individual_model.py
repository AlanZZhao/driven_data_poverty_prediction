from src.data.processed_data import a_i_test, a_h_test, b_i_test, b_h_test, c_i_test, c_h_test

from src.data.processed_data import aX_h_train, ay_h_train, bX_h_train, by_h_train, cX_h_train, cy_h_train
from src.data.processed_data import aX_i_train, ay_i_train, bX_i_train, by_i_train, cX_i_train, cy_i_train

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.externals import joblib
from pathlib import Path
import os

MODEL_DIR = os.path.join(os.path.dirname(Path(__file__).parents[1]), 'models/')

aX_i_train = aX_i_train.iloc[:,0:2]

# set generic Gradient Boosting as classifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0)

model_a_cccv=CalibratedClassifierCV(model, cv=3)

model_a_cccv.fit(aX_i_train, ay_i_train)
"""
model_b_cccv=CalibratedClassifierCV(model, cv=3)

model_b_cccv.fit(bX_i_train, by_i_train)

model_c_cccv=CalibratedClassifierCV(model, cv=3)

model_c_cccv.fit(cX_i_train, cy_i_train)
"""

# persist models
joblib.dump(model_a_cccv, os.path.join(MODEL_DIR, 'individual_a_cccv_cont.pkl'))
"""
joblib.dump(model_b_cccv, os.path.join(MODEL_DIR, 'individual_b_cccv.pkl'))
joblib.dump(model_c_cccv, os.path.join(MODEL_DIR, 'individual_c_cccv.pkl'))
"""
