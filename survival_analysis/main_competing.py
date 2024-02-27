# Author: Patricia A. Apellániz & Juan Parras
# Email: patricia.alonsod@upm.es
# Date: 27/02/2024

# Packages to import
import os
import sys
import torch
import pickle

import numpy as np

from tabulate import tabulate
from colorama import Fore, Style
from scipy.stats import ttest_ind
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from utils import run_args, load_dataset
from models_utils import train_and_evaluate_methods


def main():
    print('\n\n-------- COMPETING RISKS SURVIVAL ANALYSIS - CR-SAVAE & DEEPHIT --------')

    # Environment configuration
    seed = 1234
    rng = np.random.default_rng(seed)
    _ = torch.manual_seed(seed)
    args = run_args()

    for dataset in args['datasets']:
        ddir = os.path.join(args['output_dir'], dataset)
        if not os.path.exists(ddir):
            os.makedirs(ddir)
        if args['train']:
            df, feat_distribution, time_dist, max_t, labels, cuts = load_dataset(dataset, args, seed=seed)

            # Print some information about the dataset
            print('\n\nDataset: ' + Fore.CYAN + dataset + Style.RESET_ALL)
            print('Number of samples:', len(df))
            print('Number of features:', len(feat_distribution))
            print('Number of events:', len(labels))

            # Print the proportion of patients for each event
            for i in range(len(labels) + 1):
                print('Proportion of patients with event', i, ':', np.mean(df['event'] == i))

            # Implement 5-fold cross-validation
            df_train_folds = []
            df_val_folds = []
            kf = KFold(n_splits=args['n_folds'], shuffle=True, random_state=seed)
            for train_index, test_index in kf.split(df):
                df_train, df_val = df.iloc[train_index], df.iloc[test_index]
                df_train_folds.append(df_train)
                df_val_folds.append(df_val)

            # Parallelize the training and evaluation of the methods
            results = Parallel(n_jobs=args['n_threads'])(
                delayed(train_and_evaluate_methods)(df_train, df_val, feat_distribution, time_dist, max_t, labels, cuts,
                                                    dir=ddir) for df_train, df_val in zip(df_train_folds, df_val_folds))

            # Save results as pickle
            pickle.dump(results, open(os.path.join(ddir, 'results.pkl'), 'wb'))
        else:
            results = pickle.load(open(os.path.join(ddir, 'results.pkl'), 'rb'))

        # Print results
        print('\n\n\n----RESULTS: ' + Fore.GREEN + dataset.upper() + ' ----' + Style.RESET_ALL)
        df, feat_distribution, time_dist, max_t, labels, cuts = load_dataset(dataset, args, seed=seed)
        print('Number of samples:', len(df))
        print('Number of features:', len(feat_distribution))
        print('Number of events:', len(labels))

        # Print the proportion of patients for each event
        for i in range(len(labels) + 1):
            print('Proportion of patients with event', i, ':', np.mean(df['event'] == i))

        tab = []
        for risk in range(len(results[0])):
            for method in results[0][risk].keys():
                tab_row = [risk, 'SAVAE' if method == 'savae' else 'DeepHit']
                for metric in results[0][risk][method].keys():
                    vals = np.array([results[i][risk][method][metric] for i in range(len(results))])
                    if metric == 'ci':  # If CI < 0.5, make it 1-CI
                        vals = np.where(vals < 0.5, 1 - vals, vals)
                    if metric == 'ibrier' or metric == 'ci':  # As we account for the integrated BS, do not use the BS (redundant)
                        tab_row.append(str(np.round(np.mean(vals), 4)) + ' \pm ' + str(np.round(np.std(vals), 4)))
                tab.append(tab_row)
        print('\n')
        print(tabulate(tab, headers=['Risk', 'Method'] + list(results[0][0]['savae'].keys()), tablefmt='pretty'))
        # print(tabulate(tab, headers=['Risk', 'Method'] + list(results[0][0]['savae'].keys()), tablefmt='latex'))

        # Now, prepare the results for the t-test
        if args['test']:
            tab = []
            for risk in range(len(results[0])):
                vals = {}
                for method in results[0][risk].keys():
                    vals[method] = {}
                    for metric in results[0][risk][method].keys():
                        vals[method][metric] = np.array([results[i][risk][method][metric] for i in range(len(results))])

                # Now, perform the t-test across methods, checking whether savae is better than each other method
                for method in results[0][risk].keys():
                    if method == 'savae':
                        pass
                    else:
                        for metric in results[0][risk][method].keys():
                            if metric == 'ci':
                                # If CI < 0.5, make it 1-CI
                                vals_savae = np.where(vals['savae'][metric] < 0.5, 1 - vals['savae'][metric],
                                                      vals['savae'][metric])
                                vals_method = np.where(vals[method][metric] < 0.5, 1 - vals[method][metric],
                                                       vals[method][metric])
                                _, p = ttest_ind(vals_savae, vals_method, alternative='two-sided', equal_var=False)
                                savae_is_better = np.mean(vals_savae) > np.mean(vals_method)  # In CI, higher is better
                            else:  # Integrated IBS
                                _, p = ttest_ind(vals['savae'][metric], vals[method][metric], alternative='two-sided',
                                                 equal_var=False)
                                savae_is_better = np.mean(vals['savae'][metric]) < np.mean(
                                    vals[method][metric])  # In IBS and iIBS, lower is better
                            if p < args['significance_threshold'] and savae_is_better and metric != 'brier':
                                tab.append([risk, metric, 'BETTER', p])
                            elif p < args['significance_threshold'] and not savae_is_better and metric != 'brier':
                                tab.append([risk, metric, 'WORSE', p])
                            elif metric != 'brier':
                                tab.append([risk, metric, 'TIED', p])
                            else:
                                pass  # Do not show brier (redundant)

            print('\n')
            print(tabulate(tab, headers=['Risk', 'Metric', 'SAVAE is', 'p-value'], tablefmt='pretty'))
            # print(tabulate(tab, headers=['Risk', 'Metric', 'SAVAE is', 'p-value'], tablefmt='latex'))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
