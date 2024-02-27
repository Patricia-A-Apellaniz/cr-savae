# Author: Juan Parras & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 27/02/2024


# Packages to import
import torch

import numpy as np
import pandas as pd

from pycox.evaluation import EvalSurv
from statsmodels.stats.proportion import proportion_confint

# This warning type is removed due to pandas future warnings
# https://github.com/havakv/pycox/issues/162. Incompatibility between pycox and pandas' new version
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from base_model.vae_model import VariationalAutoencoder
from base_model.vae_utils import check_nan_inf, sample_from_dist
from base_model.vae_modules import Decoder, LogLikelihoodLossWithCensoringCompeting


# Validation-related function: used to obtain a confidence-interval during training (not the final one)
def bern_conf_interval(n, mean, ibs=False):
    # Confidence interval
    ci_bot, ci_top = proportion_confint(count=mean * n, nobs=n, alpha=0.1, method='beta')
    if mean < 0.5 and not ibs:
        ci_bot_2 = 1 - ci_top
        ci_top = 1 - ci_bot
        ci_bot = ci_bot_2
        mean = 1 - mean

    return np.round(ci_bot, 4), mean, np.round(ci_top, 4)


# Validation function: used to obtain C-Index during training (the final one is obtained in other function, to ensure
# that all methods are validated using the exactly same functions)
def obtain_c_index(surv_f, time, censor):
    # Evaluate using PyCox c-index
    ev = EvalSurv(surv_f, time.flatten(), censor.flatten(), censor_surv='km')
    ci = ev.concordance_td()

    # Obtain also ibs
    time_grid = np.linspace(time.min(), time.max(), 100)
    ibs = ev.integrated_brier_score(time_grid)
    return ci, ibs


class SAVAE_competing(VariationalAutoencoder):
    """
        Module implementing competing risks SAVAE
    """

    def __init__(self, params):
        # Initialize Savae parameters and modules
        super(SAVAE_competing, self).__init__(params)

        # Parameters
        self.time_dists = params['time_dist']
        self.max_ts = [2 * t for t in params['max_t']]
        self.n_times = len(self.time_dists)  # Number of time distributions (i.e., competing risks)

        # Add another decoder to architecture (time prediction module)

        self.Time_Decoders = [Decoder(latent_dim=self.latent_dim,
                                      hidden_size=self.hidden_size,
                                      feat_dists=[self.time_dists[i]],
                                      max_k=self.max_ts[i]) for i in range(self.n_times)]
        self.time_probs = Decoder(latent_dim=self.latent_dim,
                                  hidden_size=self.hidden_size,
                                  feat_dists=[('categorical',
                                               self.n_times)])  # Estimator of the probability of occurence of each risk

        # Define losses
        self.time_loss = LogLikelihoodLossWithCensoringCompeting(self.time_dists)

    def feed_forward(self, input_data):
        latent_params, z, cov_params = self(input_data)
        time_params = [self.Time_Decoders[i](z) for i in range(self.n_times)]
        risk_probs = self.time_probs(z)
        check_nan_inf(z, 'Time Decoder')
        out = {'z': z, 'cov_params': cov_params, 'time_params': time_params, 'latent_params': latent_params,
               'risk_probs': risk_probs}
        return out

    def predict_time(self, x, device=torch.device('cpu')):
        cov = x.drop(['time', 'event'], axis=1)
        latent_params, z, cov_params = self.predict(cov, device)
        time_params = [self.Time_Decoders[i](z) for i in range(self.n_times)]
        risk_probs = self.time_probs(z).detach().cpu().numpy()

        # Sample covariate and time values
        cov_params = cov_params.detach().cpu().numpy()
        cov_samples = sample_from_dist(cov_params, self.feat_distributions)
        time_params = [time_params[i].detach().cpu().numpy() for i in range(self.n_times)]
        time_samples = [sample_from_dist(time_params[i], [self.time_dists[i]]) for i in range(self.n_times)]

        out_data = {'z': z.detach().cpu().numpy(),
                    'cov_params': cov_params,
                    'cov_samples': cov_samples,
                    'time_params': time_params,
                    'time_samples': time_samples,
                    'risk_probs': risk_probs,
                    'latent_params': [l.detach().cpu().numpy() for l in latent_params]}

        return out_data

    def calculate_risk(self, time_train, x_val, censor_val):
        # Calculate risk (CIF) at the end of batch training
        times = np.unique(time_train)
        out_data = self.predict_time(x_val)
        pred_risk_all = []
        ci_conf_all = []
        ibs_conf_all = []
        for i in range(self.n_times):  # For each risk, note that we are computing CIFs
            pred_risk = np.zeros([out_data['time_params'][i].shape[0], len(times)])
            for sample in range(pred_risk.shape[0]):
                if self.time_dists[i][0] == 'weibull':
                    alpha = out_data['time_params'][i][sample, 0]
                    lam = out_data['time_params'][i][sample, 1]
                    prob = out_data['risk_probs'][sample, i]
                    pred_risk[sample, :] = (1 - np.exp(-np.power(times / lam, alpha))) * prob
                else:
                    raise RuntimeError('Unknown time distribution to compute risk')

            # Compute Cause-Specific C-index and IBS
            ci, ibs = obtain_c_index(pd.DataFrame(1 - pred_risk.T, index=times), np.array(x_val['time']), censor_val)
            ci_confident_intervals = bern_conf_interval(len(np.array(x_val['time'])), ci)
            ibs_confident_intervals = bern_conf_interval(len(np.array(x_val['time'])), ibs, ibs=True)

            # Save data
            pred_risk_all.append(pred_risk)
            ci_conf_all.append(ci_confident_intervals)
            ibs_conf_all.append(ibs_confident_intervals)

        return ci_conf_all, ibs_conf_all, pred_risk_all

    def fit_epoch(self, data, optimizer, batch_size=64, device=torch.device('cpu')):
        epoch_results = {'loss_tr': 0.0, 'loss_va': 0.0, 'kl_tr': 0.0, 'kl_va': 0.0, 'll_cov_tr': 0.0, 'll_cov_va': 0.0,
                         'll_time_tr': 0.0, 'll_time_va': 0.0, 'ci_va': 0.0, 'ibs_va': 0.0}
        cov_train, mask_train, time_train, censor_train, cov_val, mask_val, time_val, censor_val = data

        # Train epoch
        cov_val = torch.from_numpy(cov_val).to(device).float()
        mask_val = torch.from_numpy(mask_val).to(device).float()
        time_val = torch.from_numpy(time_val).to(device).float()
        censor_val = torch.from_numpy(censor_val).to(device).float()
        n_batches = int(np.ceil(cov_train.shape[0] / batch_size).item())
        for batch in range(n_batches):
            # Get (X, y) of the current mini batch/chunk
            index_init = batch * batch_size
            index_end = min(((batch + 1) * batch_size, cov_train.shape[
                0]))  # Use the min to prevent errors due to samples being smaller than batch_size
            cov_train_batch = cov_train[index_init: index_end]
            mask_train_batch = mask_train[index_init: index_end]
            time_train_batch = time_train[index_init: index_end]
            censor_train_batch = censor_train[index_init: index_end]

            self.train()
            cov_train_batch = torch.from_numpy(cov_train_batch).to(device).float()
            mask_train_batch = torch.from_numpy(mask_train_batch).to(device).float()
            time_train_batch = torch.from_numpy(time_train_batch).to(device).float()
            censor_train_batch = torch.from_numpy(censor_train_batch).to(device).float()

            # Generate output params
            out = self.feed_forward(cov_train_batch)

            # Compute losses
            optimizer.zero_grad()
            loss_kl = self.latent_space.kl_loss(out['latent_params'])
            loss_cov = self.rec_loss(out['cov_params'], cov_train_batch, mask_train_batch)
            loss_time = self.time_loss(out['time_params'], out['risk_probs'], time_train_batch, censor_train_batch)
            loss = loss_kl + loss_cov + loss_time
            loss.backward()
            optimizer.step()

            # Save data
            epoch_results['loss_tr'] += loss.item()
            epoch_results['kl_tr'] += loss_kl.item()
            epoch_results['ll_cov_tr'] += loss_cov.item()
            epoch_results['ll_time_tr'] += loss_time.item()

            # Validation step
            self.eval()
            with torch.no_grad():
                out = self.feed_forward(cov_val)
                loss_kl = self.latent_space.kl_loss(out['latent_params'])
                loss_cov = self.rec_loss(out['cov_params'], cov_val, mask_val)
                loss_time = self.time_loss(out['time_params'], out['risk_probs'], time_val, censor_val)
                loss = loss_kl + loss_cov + loss_time

                # Save data
                epoch_results['loss_va'] += loss.item()
                epoch_results['kl_va'] += loss_kl.item()
                epoch_results['ll_cov_va'] += loss_cov.item()
                epoch_results['ll_time_va'] += loss_time.item()

        if self.early_stop:
            self.early_stopper.early_stop(epoch_results['loss_va'])

        return epoch_results

    def fit(self, data, train_params):
        training_stats = {'loss_tr': [], 'loss_va': [], 'kl_tr': [], 'kl_va': [], 'll_cov_tr': [], 'll_cov_va': [],
                          'll_time_tr': [], 'll_time_va': [], 'ci_va': [], 'ibs_va': []}

        optimizer = torch.optim.Adam(self.parameters(), lr=train_params['lr'])

        epochs_ci = []
        epochs_ibs = []
        train_losses = []
        valid_losses = []
        best_model = None
        best_loss = np.inf
        best_ci = 0.0
        best_ibs = 1.0
        for epoch in range(train_params['n_epochs']):
            # Configure input data and missing data mask
            x_train, mask_train, x_val, mask_val = data
            time_train = np.array(x_train.loc[:, 'time'])
            time_val = np.array(x_val.loc[:, 'time'])
            censor_train = np.array(x_train.loc[:, 'event'])
            censor_val = np.array(x_val.loc[:, 'event'])
            cov_train = np.array(x_train.drop(['time', 'event'], axis=1))
            cov_val = np.array(x_val.drop(['time', 'event'], axis=1))
            mask_train = np.array(mask_train.drop(['time', 'event'], axis=1))
            mask_val = np.array(mask_val.drop(['time', 'event'], axis=1))
            assert mask_train.shape == cov_train.shape
            assert mask_val.shape == cov_val.shape
            ep_data = cov_train, mask_train, time_train, censor_train, cov_val, mask_val, time_val, censor_val

            epoch_results = self.fit_epoch(ep_data, optimizer, batch_size=train_params['batch_size'],
                                           device=train_params['device'])

            # Calculate metrics
            epoch_results['ci_va'], epoch_results['ibs_va'], _ = self.calculate_risk(time_train, x_val, censor_val)

            # Save training stats
            train_losses.append(epoch_results['loss_tr'])
            valid_losses.append(epoch_results['loss_va'])
            epochs_ci.append(epoch_results['ci_va'])
            epochs_ibs.append(epoch_results['ibs_va'])
            for key in epoch_results.keys():
                training_stats[key].append(epoch_results[key])

            if epoch % 50 == 0:
                print('Iteration = ', epoch,
                      '; train loss = ', '{:.2f}'.format(epoch_results['loss_tr']),
                      '; val loss = ', '{:.2f}'.format(epoch_results['loss_va']),
                      '; ll_cov_tr = ', '{:.2f}'.format(np.sum(epoch_results['ll_cov_tr'])),
                      '; ll_cov_va = ', '{:.2f}'.format(np.sum(epoch_results['ll_cov_va'])),
                      '; kl_tr = ', '{:.2f}'.format(epoch_results['kl_tr']),
                      '; kl_va = ', '{:.2f}'.format(epoch_results['kl_va']),
                      '; ll_time_tr = ', '{:.2f}'.format(np.sum(epoch_results['ll_time_tr'])),
                      '; ll_time_va = ', '{:.2f}'.format(np.sum(epoch_results['ll_time_va'])),
                      '; C-index = ',
                      '{:.4f} / {:.4f}'.format(np.mean(epoch_results['ci_va'][0]), np.mean(epoch_results['ci_va'][1])))

            # Save best model
            if epoch_results['loss_va'] < best_loss:
                best_loss = epoch_results['loss_va']
                best_model = self.state_dict()
                best_ci = epoch_results['ci_va']
                best_ibs = epoch_results['ibs_va']

            if self.early_stop and self.early_stopper.stop:
                print('[INFO] Training early stop')
                break

        # Set the best model as the current model
        self.load_state_dict(best_model)
        # Set the best ci and ibs in the last epoch of the validation, just to ensure that they are taken into account
        # when selecting the best seed after
        training_stats['ci_va'].append(best_ci)
        training_stats['ibs_va'].append(best_ibs)
        training_stats['best_val_loss'] = best_loss

        # Calculate CIF at the end of training
        _, _, training_stats['final_cif'] = self.calculate_risk(time_train, x_val, censor_val)

        return training_stats
