# Author: Patricia A. Apellániz & Juan Parras
# Email: patricia.alonsod@upm.es
# Date: 27/02/2024


# Packages to import
import os

import numpy as np
import pandas as pd


def cr_time_config(df):
    # Erase all rows with nan values
    df = df.dropna()

    labels = list(np.unique(df['event']))
    labels.remove(0)
    assert len(labels) > 1  # To ensure a competing risks environment: we assume that labels are 1, 2, 3, ...,
    # and 0 is censoring
    # Get the maximum time when label is not 0
    max_t = [df[df['event'] == l]['time'].max() for l in labels]
    time_dist = [('weibull', 2) for _ in range(len(labels))]
    cuts = np.unique(df['time'])
    return df, time_dist, max_t, labels, cuts


def load_dataset(dataset_name, args, seed=1234):
    if dataset_name == 'mgus2':
        # Extracted and preprocessed from https://cran.r-project.org/web/packages/survival/vignettes/survival.pdf
        df = pd.read_csv(args['input_dir'] + 'mgus2.csv')

        # DATA PREPROCESSING
        # Create the event column: if pstat == 1, event == 1, if pstat == 0, event == 2 * death,
        # where death is another column
        df['event'] = df.apply(lambda x: 1 if x['pstat'] == 1 else 2 * x['death'],
                               axis=1)  # Label meaning: 0 censoring, 1 pstat, 2 death
        # Create the time column: if pstat == 1, time == ptime, if pstat == 0, time == death
        df['time'] = df.apply(lambda x: x['ptime'] if x['pstat'] == 1 else x['futime'], axis=1)
        # Drop the columns not needed: id, ptime, pstat, futime, death
        df = df.drop(columns=['id', 'ptime', 'pstat', 'futime', 'death'])
        # Divide time by its mean to avoid numerical problems
        df['time'] = df['time'] / df['time'].mean()
        # Preprocess each covariate
        # Standardize age, hgb, creat, and mspike
        df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()
        df['hgb'] = (df['hgb'] - df['hgb'].mean()) / df['hgb'].std()
        df['creat'] = (df['creat'] - df['creat'].mean()) / df['creat'].std()
        df['mspike'] = (df['mspike'] - df['mspike'].mean()) / df['mspike'].std()
        # Replace: sex == F for 0, sex == M for 1
        df['sex'] = df['sex'].apply(lambda x: 0 if x == 'F' else 1)

        # Time and label info
        df, time_dist, max_t, labels, cuts = cr_time_config(df)
        feat_distribution = [('gaussian', 2), ('bernoulli', 1), ('gaussian', 2), ('gaussian', 2), ('gaussian', 2)]

    elif dataset_name == 'melanoma':
        # Extracted from https://github.com/tagteam/riskRegression/blob/master/prepareData/Melanoma.csv, description
        # in https://cran.r-project.org/web/packages/boot/boot.pdf
        df = pd.read_csv(args['input_dir'] + 'melanoma.csv', sep=';')

        # DATA PREPROCESSING
        # dc is death cause: 1 if death from melanoma, 2 if patient is alive, 3 if died from other cause.
        # So 2 is censoring
        df['event'] = df.apply(lambda x: 0 if x['dc'] == 2 else (1 if x['dc'] == 1 else 2),
                               axis=1)  # Label meaning: 0 censoring, 1 melanoma, 2 other cause
        df['time'] = df['days']
        # Drop the columns not needed: dc, days
        df = df.drop(columns=['dc', 'days'])
        # Divide time by its mean to avoid numerical problems
        df['time'] = df['time'] / df['time'].mean()
        # Preprocess each covariate
        # In "ecel" column, replace 2 by 0 to make it a binary in {0,1}
        df['ecel'] = df['ecel'].apply(lambda x: 0 if x == 2 else 1)
        # Standardize thick, age
        df['thick'] = (df['thick'] - df['thick'].mean()) / df['thick'].std()
        df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()

        # Time and label info
        df, time_dist, max_t, labels, cuts = cr_time_config(df)
        feat_distribution = [('categorical', 3), ('categorical', 4), ('bernoulli', 1), ('bernoulli', 1),
                             ('gaussian', 2), ('bernoulli', 1), ('gaussian', 2)]

    elif dataset_name == 'ebmt':
        # Extracted from https://cran.r-project.org/web/packages/mstate/mstate.pdf, it is called EBMT cause of
        # Death data
        df = pd.read_csv(args['input_dir'] + 'ebmt2.csv')

        # DATA PREPROCESSING
        # Keep a random subset of the data, as the dataset is quite large
        df = df.sample(n=1000, random_state=seed)
        # Time is already called time
        df['event'] = df['status']  # Survival status; 0 = censored; 1,...,6 = death due to the following causes:
        # Relapse (1), GvHD (2), Bacterial infections (3), Viral infections (4), Fungal infections (5), Other causes (6
        # Drop the columns not needed: id (patient ID), cod (same info as label)
        df = df.drop(columns=['id', 'cod', 'status'])
        # Divide time by its mean to avoid numerical problems
        df['time'] = df['time'] / df['time'].mean()
        # Preprocess the categorical covariates
        df['dissub'] = pd.Categorical(df['dissub']).codes
        df['match'] = pd.Categorical(df['match']).codes
        df['tcd'] = pd.Categorical(df['tcd']).codes
        df['year'] = pd.Categorical(df['year']).codes
        df['age'] = pd.Categorical(df['age']).codes

        # Time and label info
        df, time_dist, max_t, labels, cuts = cr_time_config(df)
        feat_distribution = [('categorical', 3), ('bernoulli', 1), ('categorical', 3), ('categorical', 3),
                             ('categorical', 3)]
    else:
        raise ValueError('Dataset not found')
    return df, feat_distribution, time_dist, max_t, labels, cuts


# Function that sets environment configuration
def run_args():
    args = {}

    # Data
    datasets = []
    dataset_name = 'all'
    if dataset_name == 'all':
        datasets = ['melanoma', 'mgus2', 'ebmt']
    else:
        datasets.append(dataset_name)
    args['datasets'] = datasets
    print('[INFO] Datasets: ', datasets)

    # Path
    args['abs_path'] = os.path.dirname(os.path.abspath(__file__)) + os.sep

    # Necessary configurations for task
    args['input_dir'] = args['abs_path'] + '/datasets/'

    # Training and testing configurations for savae and sota models
    args['n_threads'] = 5
    args['train'] = not True  # Whether to train the models or load them from file if they are pretrained
    args['test'] = True  # Whether to test using the p-values: note that old versions of scipy may yield an error in the
    # ttest_ind function
    args['n_folds'] = 5
    args['significance_threshold'] = 0.01

    args['output_dir'] = args['abs_path'] + 'results/'

    return args
