# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 29/07/2024

# Packages to import
import os
import pickle

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.svm import SVR
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


# -----------------------------------------------------------
#                      DIRECTORY UTILS
# -----------------------------------------------------------

# Function that creates output directories
def create_output_dir(config, task='survival_analysis'):
    if task == 'survival_analysis':
        params = str(config['params']['latent_dim']) + '_' + str(config['params']['hidden_size'])
        for dataset_name in config['datasets']:
            for scenario in config['scenarios']:
                n_nodes = 1 if scenario == 'centralized' else 3  # Number of nodes fixed!
                for node in range(n_nodes):
                    for seed in range(config['n_seeds']):
                        model_path = params + os.sep + 'seed_' + str(seed)
                        for mode in ['iso', 'fed_avg', 'fed_syn_naive', 'fed_syn_bias']:
                            if scenario == 'centralized' and mode != 'iso':
                                continue
                            os.makedirs(
                                config['output_path'] + dataset_name + os.sep + scenario + os.sep + 'node_' + str(
                                    node) + os.sep + model_path + os.sep + mode + os.sep, exist_ok=True)
    elif task == 'fed_syn_data_gen':
        params = str(config['params']['latent_dim']) + '_' + str(config['params']['hidden_size'])
        for dataset_name in config['datasets']:
            gen_technique = config['gen_data_technique']
            for scenario in config['scenarios']:
                n_nodes = 1 if scenario == 'centralized' else 3  # Number of nodes fixed!
                for m in ['naive', 'bias']:
                    for node in range(n_nodes):
                        if node == 0:
                            for seed in range(config['n_seeds']):
                                model_path = params + os.sep + 'seed_' + str(seed)
                                for gen_seed in range(config['n_seeds']):
                                    os.makedirs(
                                        config['gen_data_path'] + dataset_name + os.sep + scenario + os.sep + m + os.sep + 'node_' + str(
                                            node) + os.sep + gen_technique + os.sep + model_path + os.sep + 'gen_seed_' + str(
                                            gen_seed) + os.sep, exist_ok=True)

    elif task == 'imp':
        n_nodes = 2
        scenario = 'scenario_7'
        gen_technique = config['gen_data_technique']
        params = str(config['params']['latent_dim']) + '_' + str(config['params']['hidden_size'])
        for dataset_name in config['datasets']:
            col = config['pred_cols'][dataset_name]
            for node in range(n_nodes):
                for seed in range(config['n_seeds']):
                    model_path = params + os.sep + 'seed_' + str(seed)
                    for mode in ['iso', 'fed_imp', 'fed_imp_syn_naive', 'fed_imp_syn_bias']:
                        os.makedirs(config[
                                        'output_path'] + dataset_name + os.sep + scenario + os.sep + col + os.sep + 'node_' + str(
                            node) + os.sep + model_path + os.sep + mode + os.sep, exist_ok=True)
                        if node == 0 and mode != 'iso':
                            for gen_seed in range(config['n_seeds']):
                                os.makedirs(config[
                                                'gen_data_path'] + dataset_name + os.sep + scenario + os.sep + col + os.sep + mode + os.sep + 'node_' + str(
                                    node) + os.sep + gen_technique + os.sep + model_path + os.sep + 'gen_seed_' + str(
                                    gen_seed) + os.sep, exist_ok=True)

    else:
        raise NotImplementedError('Task not recognized')


# Save dictionary to pickle file
def save(res, path):
    with open(path, 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Check if file exists
def check_file(path, msg, csv=False):
    if os.path.exists(path):
        if csv:
            return pd.read_csv(path)
        else:
            file = open(path, 'rb')
            results = pickle.load(file)
            file.close()
            return results
    else:
        raise RuntimeError(msg)


def run_config(imp=False):
    config = {}

    # Data
    datasets = []
    dataset_name = 'all'  # Could be 'all' to test all cases
    if dataset_name == 'all':
        datasets = ['metabric', 'gbsg']
    else:
        datasets.append(dataset_name)
    config['datasets'] = datasets
    print('[INFO] Datasets to be preprocess: ', datasets)

    # Paths
    abs_path = os.getcwd() + os.sep
    # Remove src directory from abs_path if present (this has to do with incompatibility running from command line)
    if 'src' in abs_path:
        abs_path = abs_path.replace('src' + os.sep, '')
    config['input_path'] = abs_path + 'data' + os.sep + 'input_data' + os.sep

    # SA Training modes
    config['train'] = not True
    config['show_results'] = True  # Show results
    config['seed_eval'] = 5
    config['pv_th'] = 0.05

    # SA Training parameters
    config['n_seeds'] = 10
    config['n_threads'] = 10  # In practice, we do not use more than n_seeds threads
    config['params'] = {'hidden_size': 50, 'latent_dim': 5}  # Fixed parameters!
    config['lr'] = 1e-3
    config['n_epochs'] = -1  # -1 to use early stopping
    config['batch_size'] = 250
    config['time_dist'] = ('weibull', 2)
    config['device'] = 'cpu'

    # Gen training parameters
    config['gen_train_vae'] = True
    config['gen_early_stop'] = True
    config['gen_n_epochs'] = 10000
    config['gen_batch_size'] = 250
    config['n_gen'] = 10000  # Higher than shared_n_gen, to account for patients bad generated that are erased on postprocessing
    config['shared_n_gen'] = 5000
    config['gen_params'] = {'hidden_size': 50, 'latent_dim': 5}  # Fixed parameters!
    config['fed_steps'] = 5
    config['gen_data_path'] = abs_path + 'data' + os.sep + 'synthetic_data' + os.sep
    config['gen_data_technique'] = 'model_avg'  # Fixed technique!

    # Validation parameters
    config['classifiers_list'] = ['RF']

    # Output path
    config['output_path'] = abs_path + 'results' + os.sep

    # Validation protocol scenarios
    if imp:
        config['iso'] = True
        config['fed_imp'] = True
        config['fed_imp_syn_naive'] = True
        config['fed_imp_syn_bias'] = True

        # Column to be removed in each dataset
        config['pred_cols'] = {'metabric': 'x8', 'gbsg': 'x4'}

    else:
        config['iso'] = True
        config['fed_avg'] = True
        config['fed_syn_naive'] = True
        config['fed_syn_bias'] = True

        # CENTRALIZED: All data in one node (centralized version or upper bound)
        # SCENARIO_1: All data equally distributed in three different nodes. Training isolated, federated average and federated (synth patients) models
        # SCENARIO_2: Data not equally distributed. One node has 60%, the other 30% and the last one 10% of the data. Training isolated, federated average and federated (synth patients) models
        # SCENARIO_3: Data not equally distributed and poor data quality. One node has 60%, the other 30% and the last one 10% of the data with missing values at random. Training isolated, federated average and federated (synth patients) models
        # SCENARIO_4: Data equally distributed in terms of number of samples in three different nodes. However, there is heterogeneity in one column. (Like Scenario_1)
        # SCENARIO_5: Data not equally distributed in terms of number of samples in three different nodes. One node has 60%, the other 30% and the last one 10% of the data. There is also heterogeneity in one column. (Like Scenario_2)
        # SCENARIO_6:  Data not equally distributed and poor data quality. One node has 60%, the other 30% and the last one 10% of the data with missing values at random.  There is also heterogeneity in one column. (Like Scenario_3)
        scenario = 'all'
        if scenario == 'all':
            config['scenarios'] = ['centralized', 'scenario_1', 'scenario_2', 'scenario_3', 'scenario_4', 'scenario_5',
                                   'scenario_6']
        else:
            config['scenarios'] = [scenario]
        print('[INFO] Scenarios: ', config['scenarios'])

    return config


# -----------------------------------------------------------
#                      DATA UTILS
# -----------------------------------------------------------

def zero_imputation(data):
    imp_data = data.copy()
    imp_data = imp_data.fillna(0)
    return imp_data


def mice_imputation(data, model='bayesian'):
    imp_data = data.copy()
    if model == 'bayesian':
        clf = BayesianRidge()
    elif model == 'svr':
        clf = SVR()
    else:
        raise RuntimeError('MICE imputation base_model not recognized')
    imp = IterativeImputer(estimator=clf, verbose=2, max_iter=30, tol=1e-10, imputation_order='roman')
    imp_data.iloc[:, :] = imp.fit_transform(imp_data)
    return imp_data


def statistics_imputation(data, raw_data, norm):
    imp_data = data.copy()
    # If data comes from classification task, columns size doesn't match data's columns size
    n_columns = data.columns.size if data.columns.size < data.columns.size else data.columns.size
    for i in range(n_columns):
        values = data.iloc[:, i].values
        raw_values = raw_data.iloc[:, i].values
        if any(pd.isnull(values)):
            no_nan_values = values[~pd.isnull(values)]
            no_nan_raw_values = raw_values[~pd.isnull(raw_values)]
            if values.dtype in [object, str] or no_nan_values.size <= 2 or np.amin(
                    np.equal(np.mod(no_nan_values, 1), 0)):
                stats_value = stats.mode(no_nan_values, keepdims=True)[0][0]
            # If raw data has int values take mode normalized
            elif norm and np.amin(np.equal(np.mod(no_nan_raw_values, 1), 0)):
                stats_value = stats.mode(no_nan_raw_values, keepdims=True)[0][0]
                # Find index of stats_value in self.raw_df.iloc[:, i].values
                idx = np.where(raw_data.iloc[:, i].values == stats_value)[0][0]
                # Find which value is in idx of data.iloc[:, i].values and set this value to stats_value
                stats_value = values[np.where(values == data.iloc[:, i].values[idx])[0][0]]
            else:
                stats_value = no_nan_values.mean()
            imp_data.iloc[:, i] = [stats_value if pd.isnull(x) else x for x in imp_data.iloc[:, i]]

    return imp_data


def impute_data(df, raw_df, mode='stats', norm=True):
    # If missing data exists, impute it
    if df.isna().any().any():
        # Data imputation
        if mode == 'zero':
            imp_df = zero_imputation(df)
        elif mode == 'stats':
            imp_df = statistics_imputation(df, raw_df, norm)
        else:
            imp_df = mice_imputation(df)
    else:
        imp_df = df.copy()

    return imp_df


def get_col_distributions(df):
    col_dist = []
    for i in range(df.shape[1]):
        values = df.iloc[:, i].unique()
        no_nan_values = values[~pd.isnull(values)]
        if no_nan_values.size <= 2:
            col_dist.append(('bernoulli', 1))
        else:
            col_dist.append(('gaussian', 2))
    # Change time distribution to weibull
    col_dist[-2] = ('weibull', 2)

    return col_dist


# Transform data according to raw_df
def transform_data(df, raw_df, col_dist, denorm=False):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    transformed_df = df.copy()
    for i in range(raw_df.shape[1]):
        dist = col_dist[i][0]
        values = raw_df.iloc[:, i]
        no_nan_values = values[~pd.isnull(values)].values
        if dist == 'gaussian':
            loc = np.mean(no_nan_values)
            scale = np.std(no_nan_values)
        elif dist == 'log-normal':
            loc = np.mean(no_nan_values)
            scale = np.std(no_nan_values)
        elif dist == 'bernoulli':
            loc = np.amin(no_nan_values)
            scale = np.amax(no_nan_values) - np.amin(no_nan_values)
        elif dist == 'categorical':
            loc = np.amin(no_nan_values)
            scale = 1  # Do not scale
        elif dist == 'weibull':
            loc = 1 if 0 in no_nan_values else 0
            scale = 0
        else:
            raise NotImplementedError('Distribution ', dist, ' not normalized!')

        if denorm:  # Denormalize
            if dist == 'weibull':
                transformed_df.iloc[:, i] = (df.iloc[:, i] - loc).astype(raw_df.iloc[:, i].dtype)
            else:
                # transformed_df.iloc[:, i] = (df.iloc[:, i] * scale + loc if scale != 0 else df.iloc[:, i] + loc).astype(raw_df.iloc[:, i].dtype)
                if raw_df.iloc[:, i].dtype == 'int64':
                    transformed_df.iloc[:, i] = (
                        df.iloc[:, i] * scale + loc if scale != 0 else df.iloc[:, i] + loc).round().astype(
                        raw_df.iloc[:, i].dtype)
                else:
                    transformed_df.iloc[:, i] = (
                        df.iloc[:, i] * scale + loc if scale != 0 else df.iloc[:, i] + loc).astype(
                        raw_df.iloc[:, i].dtype)
                # check if raw_df values are floats but have no decimal values and round transformed_df values
                if np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
                    transformed_df.iloc[:, i] = transformed_df.iloc[:, i].round()

        else:  # Normalize
            if dist == 'weibull':
                transformed_df.iloc[:, i] = df.iloc[:, i] + loc
            else:
                transformed_df.iloc[:, i] = (df.iloc[:, i] - loc) / scale if scale != 0 else df.iloc[:, i] - loc

    return transformed_df

def postprocess_gen_data(gen_info, raw_df):
    # Denormalize generated samples to check data ranges
    cov_samples = gen_info['cov_samples'].copy()

    # Transform only the time: it was Gaussian, now set it to Weibull
    mean, std = raw_df.iloc[:, -2].mean(), raw_df.iloc[:, -2].std()
    cov_samples[:, -2] = cov_samples[:, -2] * std + mean

    # Erase negative values of the time column
    cov_samples = cov_samples[cov_samples[:, -2] >= 0]

    # If the minimum time is 0, add 1 to all the times
    if cov_samples[:, -2].min() == 0:
        cov_samples[:, -2] += 1

    # Double check
    assert cov_samples[:, -2].min() > 0

    # Make cov_samples a DataFrame with raw_df columns
    cov_samples = pd.DataFrame(cov_samples, columns=raw_df.columns)

    gen_info['cov_samples'] = cov_samples

    return gen_info



def fill_sequence(sequence, n_samples):
    # First, save the last value, as that must be also the last value of the normalized sequence
    last_value = sequence[-1]
    # Normalize the sequence to have n_samples
    new_seq = -np.ones((n_samples, sequence.shape[1]))  # Use -1 as a flag value for plotting later
    new_seq[:sequence.shape[0], :] = sequence
    # Set the last value to the original last value
    new_seq[-1] = last_value
    return new_seq
