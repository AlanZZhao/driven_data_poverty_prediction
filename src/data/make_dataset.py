# -*- coding: utf-8 -*-
import os

from pathlib import Path

import numpy as np
import pandas as pd
import sys
from statsmodels.imputation import mice

# data directory
DATA_DIR = os.path.join(os.path.dirname(Path(__file__).parents[1]), 'data/raw')
OUTPUT_DATA_DIR = os.path.join(os.path.dirname(Path(__file__).parents[1]), 'data/processed')


# dictionaries created for simplicity in managing file paths
household_paths = {'A': {'train': os.path.join(DATA_DIR, 'A_hhold_train.csv'),
                         'test': os.path.join(DATA_DIR, 'A_hhold_test.csv')},

                   'B': {'train': os.path.join(DATA_DIR, 'B_hhold_train.csv'),
                         'test': os.path.join(DATA_DIR, 'B_hhold_test.csv')},

                   'C': {'train': os.path.join(DATA_DIR, 'C_hhold_train.csv'),
                         'test': os.path.join(DATA_DIR, 'C_hhold_test.csv')}}


individual_paths = {'A': {'train': os.path.join(DATA_DIR, 'A_indiv_train.csv'),
                          'test': os.path.join(DATA_DIR, 'A_indiv_test.csv')},

                    'B': {'train': os.path.join(DATA_DIR, 'B_indiv_train.csv'),
                          'test': os.path.join(DATA_DIR, 'B_indiv_test.csv')},

                    'C': {'train': os.path.join(DATA_DIR, 'C_indiv_train.csv'),
                          'test': os.path.join(DATA_DIR, 'C_indiv_test.csv')}}

def standardize(df, numeric_only=True):
    # detect columns that are numeric
    numeric = df.select_dtypes(include=['int64', 'float64'])

    # subtract mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()

    return df


def pre_process_data(df, enforce_cols=None):
    print("Input shape:\t{}".format(df.shape))


    df = standardize(df)
    print("After standardization {}".format(df.shape))

    # create dummy variables for categoricals
    df = pd.get_dummies(df)
    print("After converting categoricals:\t{}".format(df.shape))


    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})

    #df.fillna(0, inplace=True)

    return df

def impute(df, type='household', perturbation_method='gaussian', k_pmm=20, history_callback=None):
    # wrapper for impute to preserve index

    if type is 'household':
        index = df.index

    imputed_df = mice.MICEData(df, perturbation_method, k_pmm, history_callback).data

    imputed_df.set_index(index, inplace=True)
    return imputed_df


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # read data in
    a_h_train = pd.read_csv(household_paths['A']['train'], index_col='id')
    b_h_train = pd.read_csv(household_paths['B']['train'], index_col='id')
    c_h_train = pd.read_csv(household_paths['C']['train'], index_col='id')

    a_i_train = pd.read_csv(individual_paths['A']['train'], index_col=['id', 'iid'])
    b_i_train = pd.read_csv(individual_paths['B']['train'], index_col=['id', 'iid'])
    c_i_train = pd.read_csv(individual_paths['C']['train'], index_col=['id', 'iid'])

    # read test data in
    a_h_test = pd.read_csv(household_paths['A']['test'], index_col='id')
    b_h_test = pd.read_csv(household_paths['B']['test'], index_col='id')
    c_h_test = pd.read_csv(household_paths['C']['test'], index_col='id')

    a_i_test = pd.read_csv(individual_paths['A']['test'], index_col=['id', 'iid'])
    b_i_test = pd.read_csv(individual_paths['B']['test'], index_col=['id', 'iid'])
    c_i_test = pd.read_csv(individual_paths['C']['test'], index_col=['id', 'iid'])

    # run processing of data
    aX_h_train = pre_process_data(a_h_train.drop(['poor', 'country'], axis=1))
    ay_h_train = np.ravel(a_h_train.poor)

    bX_h_train = pre_process_data(b_h_train.drop(['poor', 'country'], axis=1))
    by_h_train = np.ravel(b_h_train.poor)

    cX_h_train = pre_process_data(c_h_train.drop(['poor', 'country'], axis=1))
    cy_h_train = np.ravel(c_h_train.poor)

    aX_i_train = pre_process_data(a_i_train.drop(['poor', 'country'], axis=1))
    ay_i_train = np.ravel(a_i_train.poor)

    bX_i_train = pre_process_data(b_i_train.drop(['poor', 'country'], axis=1))
    by_i_train = np.ravel(b_i_train.poor)

    cX_i_train = pre_process_data(c_i_train.drop(['poor', 'country'], axis=1))
    cy_i_train = np.ravel(c_i_train.poor)

    a_h_test = pre_process_data(a_h_test.drop('country', axis=1), enforce_cols=aX_h_train.columns)
    b_h_test = pre_process_data(b_h_test.drop('country', axis=1), enforce_cols=bX_h_train.columns)
    c_h_test = pre_process_data(c_h_test.drop('country', axis=1), enforce_cols=cX_h_train.columns)

    a_i_test = pre_process_data(a_i_test.drop('country', axis=1), enforce_cols=aX_i_train.columns)
    b_i_test = pre_process_data(b_i_test.drop('country', axis=1), enforce_cols=bX_i_train.columns)
    c_i_test = pre_process_data(c_i_test.drop('country', axis=1), enforce_cols=cX_i_train.columns)

    bX_h_train = impute(bX_h_train)
    b_h_test = impute(b_h_test)
    aX_i_train = impute(aX_i_train)
    a_i_test = impute(a_i_test)
    bX_i_train = impute(bX_i_train)
    b_i_test = impute(b_i_test)

    # save data
    aX_h_train.to_csv(os.path.join(OUTPUT_DATA_DIR, 'aX_h_train.csv'))
    np.save(os.path.join(OUTPUT_DATA_DIR, 'ay_h_train.npy'), ay_h_train)

    bX_h_train.to_csv(os.path.join(OUTPUT_DATA_DIR, 'bX_h_train.csv'))
    np.save(os.path.join(OUTPUT_DATA_DIR, 'by_h_train.npy'), by_h_train)

    cX_h_train.to_csv(os.path.join(OUTPUT_DATA_DIR, 'cX_h_train.csv'))
    np.save(os.path.join(OUTPUT_DATA_DIR, 'cy_h_train.npy'), cy_h_train)

    aX_i_train.to_csv(os.path.join(OUTPUT_DATA_DIR, 'aX_i_train.csv'))
    np.save(os.path.join(OUTPUT_DATA_DIR, 'ay_i_train.npy'), ay_i_train)

    bX_i_train.to_csv(os.path.join(OUTPUT_DATA_DIR, 'bX_i_train.csv'))
    np.save(os.path.join(OUTPUT_DATA_DIR, 'by_i_train.npy'), by_i_train)

    cX_i_train.to_csv(os.path.join(OUTPUT_DATA_DIR, 'cX_i_train.csv'))
    np.save(os.path.join(OUTPUT_DATA_DIR, 'cy_i_train.npy'), cy_i_train)

    a_h_test.to_csv(os.path.join(OUTPUT_DATA_DIR, 'a_h_test.csv'))
    b_h_test.to_csv(os.path.join(OUTPUT_DATA_DIR, 'b_h_test.csv'))
    c_h_test.to_csv(os.path.join(OUTPUT_DATA_DIR, 'c_h_test.csv'))

    a_i_test.to_csv(os.path.join(OUTPUT_DATA_DIR, 'a_i_test.csv'))
    b_i_test.to_csv(os.path.join(OUTPUT_DATA_DIR, 'b_i_test.csv'))
    c_i_test.to_csv(os.path.join(OUTPUT_DATA_DIR, 'c_i_test.csv'))


if __name__ == '__main__':
    main()
