# Author: Juan Parras & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 27/02/2024

import os
import torch
import numpy as np
import pandas as pd
import torchtuples as tt
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
# import tikzplotlib  # Due to an error, this requires matplotlib <= 3.7.0 (hopefully, it will be fixed in future versions)

from pycox.models import DeepHit
from pycox.evaluation import EvalSurv
from cr_savae import SAVAE_competing


class CauseSpecificNet(torch.nn.Module):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """

    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                 out_features, batch_norm=True, dropout=None):
        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features, num_nodes_shared[:-1], num_nodes_shared[-1],
            batch_norm, dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net = tt.practical.MLPVanilla(
                num_nodes_shared[-1], num_nodes_indiv, out_features,
                batch_norm, dropout,
            )
            self.risk_nets.append(net)

    def forward(self, input):
        out = self.shared_net(input)
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)
        return out


def train_deephit(df_train, df_val, cuts, epochs=512, batch_size=256, verbose=True, num_nodes_shared=[64, 64],
                  num_nodes_indiv=[32], batch_norm=True, dropout=0.1):  # Interface for DeepHit
    """
    Train DeepHit
    :param df_train: Dataframe with covariates (any name), time and event (must be named 'time' and 'event')
    :param df_val: Dataframe with covariates (any name), time and event (must be named 'time' and 'event')
    :param cuts: List of time cuts for the training
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param verbose: Whether to print training information
    :param num_nodes_shared: Number of nodes in the shared layers
    :param num_nodes_indiv: Number of nodes in the individual layers
    :param batch_norm: Whether to use batch normalization
    :param dropout: Dropout rate
    :return: Trained model and training results
    """
    x_train = df_train.drop(columns=['time', 'event']).values.astype('float32')
    x_val = df_val.drop(columns=['time', 'event']).values.astype('float32')

    y_train = (df_train['time'].values.astype('int64'), df_train['event'].values.astype('int64'))
    y_val = (df_val['time'].values.astype('int64'), df_val['event'].values.astype('int64'))
    val = (x_val, y_val)

    in_features = x_train.shape[1]
    num_risks = y_train[1].max()
    out_features = len(cuts)

    net = CauseSpecificNet(in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                           out_features, batch_norm, dropout)

    optimizer = tt.optim.AdamWR(lr=0.01, decoupled_weight_decay=0.01, cycle_eta_multiplier=0.8)
    model = DeepHit(net, optimizer, alpha=0.2, sigma=0.1, duration_index=cuts)

    callbacks = [tt.callbacks.EarlyStoppingCycle()]

    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val)
    return model, log


def train_savae(df_train, df_val, feat_distribution, time_distribution, max_t, n_seeds=1, log_name='./run',
                latent_dim=5, hidden_size=64, epochs=2000, batch_size=32, lr=1e-3,
                early_stop=True):  # Interface for SAVAE
    """
    Train SAVAE with competing risks
    :param df_train: Dataframe with covariates (any name), time and event (must be named 'time' and 'event')
    :param df_val: Dataframe with covariates (any name), time and event (must be named 'time' and 'event')
    :param feat_distribution: List of tuples with the distribution of each covariate, and the number of parameters of the distribution. Example: [('gaussian', 2), ('weibull', 2), ('categorical', 5)]
    :param time_distribution: List of tuples with the distribution of each cause of failure, and the number of parameters of the distribution. Example: [('weibull', 2), ('weibull', 2), ('weibull', 2)]
    :param max_t: Maximum time for each cause of failure
    :param n_seeds: Number of seeds to train the model
    :param log_name: Name of the log file
    :param latent_dim: Dimension of the latent space
    :param hidden_size: Number of nodes in the hidden layers
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param lr: Learning rate
    :param early_stop: Whether to use early stopping
    :return: Trained model and training results
    """
    model_params = {'feat_distributions': feat_distribution,
                    'latent_dim': latent_dim,
                    'hidden_size': hidden_size,
                    'input_dim': len(feat_distribution),
                    'max_t': max_t,
                    'time_dist': time_distribution,
                    'early_stop': early_stop}
    data = (df_train, pd.DataFrame(1, index=df_train.index, columns=df_train.columns),
            df_val, pd.DataFrame(1, index=df_val.index, columns=df_val.columns))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # Force CPU for parallelization
    print('Training SAVAE on device:', device)
    train_params = {'n_epochs': epochs, 'batch_size': batch_size, 'device': device, 'lr': lr, 'path_name': log_name}

    def train_savae_seed(data, model_params, train_params):
        model = SAVAE_competing(model_params)
        # Train model
        training_results = model.fit(data, train_params)
        return model, training_results

    out = Parallel(n_jobs=n_seeds)(delayed(train_savae_seed)(data, model_params, train_params) for _ in range(n_seeds))
    # Return model with lower validation loss
    val_losses = [r[1]['best_val_loss'] for r in out]
    best_model = out[np.argmin(val_losses)]
    return best_model[0], best_model[1]


def train_and_evaluate_methods(df_train, df_val, feat_distribution, time_dist, max_t, labels, cuts, dir='/results'):
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # Train SAVAE and DeepHit
    sa_model, sa_results = train_savae(df_train, df_val, feat_distribution, time_dist, max_t=max_t, n_seeds=1,
                                       log_name=dir)  # Run SAVAE
    dh_model, dh_results = train_deephit(df_train, df_val, cuts)  # Run DeepHit

    # Validate both methods using the validation set
    x_val = df_val.drop(columns=['time', 'event']).values.astype('float32')
    y_val = (df_val['time'].values.astype('int64'), df_val['event'].values.astype('int64'))
    cif = dh_model.predict_cif(x_val)
    _, _, cif_savae = sa_model.calculate_risk(df_train['time'].values, df_val, y_val[1])
    times = np.unique(df_train['time'])
    results = []
    for i in range(len(labels)):  # For each risk
        # Evaluate SAVAE
        cif_risk_savae = pd.DataFrame(cif_savae[i].T, index=times)
        ev_sa = EvalSurv(1 - cif_risk_savae, y_val[0], y_val[1] == i + 1, censor_surv='km')
        r_sa = {'ci': ev_sa.concordance_td(), 'brier': ev_sa.integrated_brier_score(times),
                'ibrier': ev_sa.integrated_brier_score(times)}
        # Evaluate DeepHit
        cif_risk_dh = pd.DataFrame(cif[i], dh_model.duration_index)
        ev_dh = EvalSurv(1 - cif_risk_dh, y_val[0], y_val[1] == i + 1, censor_surv='km')
        r_dh = {'ci': ev_dh.concordance_td(), 'brier': ev_dh.integrated_brier_score(times),
                'ibrier': ev_dh.integrated_brier_score(times)}
        results.append({'savae': r_sa, 'deephit': r_dh})

    # Plot the estimated curves to show results (note that these results are for a single fold if KFold is used)
    n_pats = min([10, df_val.shape[0]])  # Save evaluation results for the first 10 patients, for graphical purposes
    c = ['r', 'b', 'g', 'k', 'm', 'c']
    for i in range(n_pats):  # Patient
        for j in range(len(labels)):  # Risk
            plt.plot(cuts, cif[j, :, i], label='DH: Risk ' + str(j + 1), color=c[j], linestyle='dashed')
            plt.plot(np.unique(df_train['time'].values), cif_savae[j][i, :], label='SAVAE: Risk ' + str(j + 1),
                     color=c[j])
        plt.legend(loc='best')
        plt.xlabel('Time')
        plt.ylabel('CIF')
        plt.title('Patient ' + str(i))
        plt.savefig(os.path.join(dir, 'cif_patient_' + str(i) + '.png'))
        # tikzplotlib.save(os.path.join(dir, 'cif_patient_' + str(i) + '.tex')) # Uncomment if tikzplotlib is installed
        plt.close()

    return results
