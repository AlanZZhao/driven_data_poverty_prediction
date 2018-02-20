from src.data.interim_data import aX_h_train_resampled, ay_h_train_resampled, bX_h_train_resampled, by_h_train_resampled, cX_h_train_resampled, cy_h_train_resampled
from src.data.interim_data import a_h_test, b_h_test, c_h_test
from src.data.interim_data import aX_h_train, bX_h_train, cX_h_train
from src.data.processed_data import ay_h_train, by_h_train, cy_h_train


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals import joblib
from pathlib import Path
import os

MODEL_DIR = os.path.join(os.path.dirname(Path(__file__).parents[1]), 'models/')


rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True)

param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_features': ['auto', 'sqrt', 'log2']
}


model_b = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 3, scoring = 'neg_log_loss')
model_b.fit(bX_h_train_resampled, by_h_train_resampled)

model_c = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 3, scoring = 'neg_log_loss')
model_c.fit(cX_h_train_resampled, cy_h_train_resampled)


joblib.dump(model_b, os.path.join(MODEL_DIR, 'household_b_rf_oversample.pkl'))
joblib.dump(model_c, os.path.join(MODEL_DIR, 'household_c_rf_oversample.pkl'))
