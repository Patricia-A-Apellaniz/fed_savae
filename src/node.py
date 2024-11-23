# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 08/10/2024

# Import libraries
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from copy import deepcopy

from src.savae.savae import SAVAE
from src.synthetic_data_generation.generation_manager import generate
from src.utils import get_col_distributions, transform_data, impute_data, check_file


# Always use 1000 samples to validate
def split_data(denorm_df, col_dist, seed):
    # Shuffle data
    denorm_df = denorm_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Take all samples minus 1000 for training from the denormalize dataset. Then, normalize it based only on the training data
    denorm_train_data = denorm_df[:-1000].copy()
    train_data = transform_data(denorm_train_data, denorm_train_data, col_dist)
    denorm_test_data = denorm_df[-1000:].copy().reset_index(drop=True)
    test_data = transform_data(denorm_test_data, denorm_train_data, col_dist)

    # It could happen that test_data has 0 in time column but train_data has not.
    # In this case, we should add 1 to both datasets
    if np.any(test_data['time'] == 0) and not np.any(train_data['time'] == 0):
        train_data['time'] = train_data['time'] + 1
        test_data['time'] = test_data['time'] + 1

    # Obtain nan masks
    train_nans = train_data.isna()
    train_mask = train_nans.replace([True, False], [0, 1])
    test_nans = test_data.isna()
    test_mask = test_nans.replace([True, False], [0, 1])

    # Impute data
    train_imp_data = impute_data(train_data, denorm_train_data, mode='stats')
    test_imp_data = impute_data(test_data, denorm_test_data, mode='stats')

    # Model data
    data = (train_imp_data, train_mask, test_imp_data, test_mask)
    denorm_data = (denorm_train_data, denorm_test_data)

    return data, denorm_data


def remove_neg(df, dataset_name):
    transf_df = df.copy()
    if dataset_name == 'metabric':
        transf_df = transf_df[transf_df.loc[:, 'time'] >= 0]
    elif dataset_name == 'gbsg':
        transf_df = transf_df[transf_df.loc[:, 'x4'] >= 0]
        transf_df = transf_df[transf_df.loc[:, 'time'] >= 0]
    return transf_df


# Function that introduces missing values in the data. NOT IN TIME AND EVENT COLUMNS!!
def introduce_missing_values(df, perc):
    missing_df = df.copy()

    # Flatten the DataFrame to work with a 1D array of values
    num_rows, num_cols = missing_df.shape
    total_values = num_rows * num_cols
    num_nans = int(np.floor(total_values * perc))

    # Get a list of all indices
    indices = np.arange(total_values)

    # Randomly select indices to be replaced with NaN
    np.random.shuffle(indices)
    nan_indices = indices[:num_nans]

    # Convert the flat indices back to 2D indices
    row_indices = nan_indices // num_cols
    col_indices = nan_indices % num_cols

    # Replace the selected positions with NaN. If position is in two last columns, don't replace it
    for row, col in zip(row_indices, col_indices):
        if col < num_cols - 2:
            missing_df.iat[row, col] = np.nan

    return missing_df


# Function that splits the data into different nodes and folds. Three nodes maximum for these experiments
def add_heterogeneity(df, dataset_name, nodes_samples, seed=1234):
    # Take particular column and compute median (for example age column)
    # First node will have normal values, second node will have more values below the median and third node will have values above the median
    # For this example, I will use the age column
    col = 'age'
    if dataset_name == 'metabric':
        col = 'x8'
    elif dataset_name == 'gbsg':
        col = 'x3'  # https://warwick.ac.uk/fac/sci/statistics/apts/students/resources-1617/gbcs.txt

    # Compute median
    median_age = df[col].median()

    # Take samples not appearing in node 0 and divide them into two groups: below and above the median
    nodes_1_2_df = df.iloc[nodes_samples[1]:nodes_samples[0] + nodes_samples[1] + nodes_samples[2]].reset_index(
        drop=True)
    below_median = nodes_1_2_df[nodes_1_2_df[col] <= median_age]
    above_median = nodes_1_2_df[nodes_1_2_df[col] > median_age]

    # Node 1
    # Take n_samples as follows: a 95% of the samples should have age below the median and the rest above the median, having a total of n_samples
    samples_below = below_median.sample(n=int(0.95 * nodes_samples[1]), random_state=seed)
    samples_above = above_median.sample(n=int(np.ceil(0.05 * nodes_samples[1])), random_state=seed)
    no_iid_1_df = pd.concat([samples_below, samples_above]).reset_index(drop=True)

    # If elements in nodes_samples are equal, then node 2 will have the same samples as node 1
    if nodes_samples[0] == nodes_samples[1] == nodes_samples[2]:
        no_iid_2_df = nodes_1_2_df[~nodes_1_2_df.index.isin(no_iid_1_df.index)].reset_index(drop=True)
    else:
        # Node 2
        # Remove samples from node 1 in below_median and above_median
        below_median = below_median[~below_median.index.isin(samples_below.index)]
        above_median = above_median[~above_median.index.isin(samples_above.index)]
        # Take n_samples as follows: a 95% of the samples should have age above the median and the rest below the median, having a total of n_samples
        samples_above = above_median.sample(n=int(0.95 * nodes_samples[2]), random_state=seed)
        samples_below = below_median.sample(n=int(np.ceil(0.05 * nodes_samples[2])), random_state=seed)
        no_iid_2_df = pd.concat([samples_above, samples_below])

    return no_iid_1_df, no_iid_2_df, median_age, col


# This scenario has just 2 node with the same number of samples. However, an important column has been removed from the second node
def manage_scenario_7_data(df, real_df, dataset, config, seed=1234):
    scenario_data = []
    col_dist = get_col_distributions(df)

    # Denormalize df based on original df. Each node should normalize its data based on its own data, not the whole dataset. We are simulating a real scenario!!
    gen_col_dist = col_dist.copy()
    gen_col_dist[-2] = ('gaussian', 2)
    denorm_df = transform_data(df, real_df, gen_col_dist, denorm=True)
    transf_denorm_df = remove_neg(denorm_df, dataset)

    # Now, we have the 10.000 raw samples. We can forget about the original_df!
    # I will just use 3000 samples for each node. Then, I need a total of 9000 samples for the whole dataset
    total_denorm_samples_df = transf_denorm_df.sample(n=9000, random_state=seed).reset_index(drop=True)

    # Split data into n_nodes
    n_nodes = 2
    n_samples = total_denorm_samples_df.shape[0]
    samples_per_node = n_samples // n_nodes
    col = config['pred_cols'][dataset]

    # This scenario has 2 nodes with the same number of samples. In the first node, all columns are complete.
    # In the second node, we will remove the most important column.

    # Split data into n_nodes
    for node in range(n_nodes):
        denorm_node_df = total_denorm_samples_df.iloc[node * samples_per_node:(node + 1) * samples_per_node]

        # Split data
        node_data, denorm_node_data = split_data(denorm_node_df, col_dist, seed)

        if node == 1:
            train_node_df, train_node_mask, test_node_df, test_node_mask = node_data
            denorm_train_node_df, denorm_test_node_df = denorm_node_data

            # Remove col column from node data
            idx = train_node_df.columns.get_loc(col)
            train_node_col_drop_df = train_node_df.drop(columns=[col])
            train_node_col_drop_mask = train_node_mask.drop(columns=[col])
            test_node_col_drop_df = test_node_df.drop(columns=[col])
            test_node_col_drop_mask = test_node_mask.drop(columns=[col])
            denorm_train_node_col_drop_df = denorm_train_node_df.drop(columns=[col])
            denorm_test_node_col_drop_df = denorm_test_node_df.drop(columns=[col])
            node_col_drop_data = (train_node_col_drop_df, train_node_col_drop_mask, test_node_col_drop_df, test_node_col_drop_mask)
            denorm_node_col_drop_data = (denorm_train_node_col_drop_df, denorm_test_node_col_drop_df)

            # Remove item from col idx in col_dist
            node_1_col_dist = col_dist.copy()
            del node_1_col_dist[idx]

            # In this scenario we are going to validate using test data with pred col and test data with original col
            node_no_col_drop_test_data = (test_node_df, test_node_mask)

            scenario_data.append(Node(node_col_drop_data, denorm_node_col_drop_data, node_1_col_dist, dataset, 'scenario_7', config, 'node_' + str(node), no_col_drop_test_data=node_no_col_drop_test_data))

        else:
            scenario_data.append(Node(node_data, denorm_node_data, col_dist, dataset, 'scenario_7', config, 'node_' + str(node)))

    return scenario_data


def manage_data(df, original_df, scenario, dataset_name, config, n_nodes=3, col_dist=None, seed=42):
    scenario_data = []
    col_dist = get_col_distributions(df) if col_dist is None else col_dist

    # Denormalize df based on original df. Each node should normalize its data based on its own data, not the whole dataset. We are simulating a real scenario!!
    gen_col_dist = col_dist.copy()
    gen_col_dist[-2] = ('gaussian', 2)
    denorm_df = transform_data(df, original_df, gen_col_dist, denorm=True)

    # Remove negative values
    transf_denorm_df = remove_neg(denorm_df, dataset_name)

    # Now, we have the 10.000 raw samples. We can forget about the original_df!
    # I will just use 3000 samples for each node. Then, I need a total of 9000 samples for the whole dataset
    total_denorm_samples_df = transf_denorm_df.sample(n=9000, random_state=seed).reset_index(drop=True)

    # Split data into n_nodes depending on the scenario
    if scenario == 'centralized':
        node_data, denorm_node_data = split_data(total_denorm_samples_df, col_dist, seed)
        scenario_data.append(Node(node_data, denorm_node_data, col_dist, dataset_name, scenario, config, 'node_0'))

    elif scenario == 'scenario_1':
        n_samples = total_denorm_samples_df.shape[0]
        samples_per_node = n_samples // n_nodes

        # Split data into n_nodes
        for node in range(n_nodes):
            denorm_node_df = total_denorm_samples_df.iloc[node * samples_per_node:(node + 1) * samples_per_node]
            node_data, denorm_node_data = split_data(denorm_node_df, col_dist, seed)
            scenario_data.append(
                Node(node_data, denorm_node_data, col_dist, dataset_name, scenario, config, 'node_' + str(node)))

    elif scenario in ['scenario_2', 'scenario_3']:
        samples_per_node = [3000, 1500, 1050]  # 2000, 750 and 100 training samples for each node

        # Split data into n_nodes
        for i in range(n_nodes):
            if i == 0:
                denorm_node_df = total_denorm_samples_df.iloc[i:(i + 1) * samples_per_node[i]]
            elif i == 1:
                denorm_node_df = total_denorm_samples_df.iloc[
                                 samples_per_node[i - 1]:samples_per_node[i - 1] + samples_per_node[i]]
            else:
                denorm_node_df = total_denorm_samples_df.iloc[samples_per_node[i - 2]
                                                              + samples_per_node[i - 1]:(samples_per_node[i - 2]
                                                                                         + samples_per_node[i - 1])
                                                                                        + samples_per_node[i]]
                # Introduce missing values
                if scenario == 'scenario_3':
                    denorm_node_df = introduce_missing_values(denorm_node_df, 0.5)
            node_data, denorm_node_data = split_data(denorm_node_df, col_dist, seed)
            scenario_data.append(
                Node(node_data, denorm_node_data, col_dist, dataset_name, scenario, config, 'node_' + str(i)))

    elif scenario in ['scenario_4', 'scenario_5', 'scenario_6']:
        if scenario == 'scenario_4':
            n_samples = total_denorm_samples_df.shape[0]
            samples_per_node = [n_samples // n_nodes, n_samples // n_nodes, n_samples // n_nodes]
        else:
            samples_per_node = [3000, 1500, 1050]  # 2000, 500 and 50 training samples for each node

        # Split data into n_nodes
        # Node 0
        denorm_node_0_df = total_denorm_samples_df.iloc[0:samples_per_node[0]]
        node_0_data, denorm_node_0_data = split_data(denorm_node_0_df, col_dist, seed)
        # Node 1 and 2
        no_iid_node_1_df, no_iid_node_2_df, median_age, age_col = add_heterogeneity(total_denorm_samples_df,
                                                                                    dataset_name, samples_per_node,
                                                                                    seed=seed)
        node_1_data, denorm_node_1_data = split_data(no_iid_node_1_df, col_dist, seed)
        # Node 2
        if scenario == 'scenario_6':
            no_iid_node_2_df = introduce_missing_values(no_iid_node_2_df, 0.5)
        node_2_data, denorm_node_2_data = split_data(no_iid_node_2_df, col_dist, seed)

        scenario_data.append(
            Node(node_0_data, denorm_node_0_data, col_dist, dataset_name, scenario, config, 'node_0', age_col=age_col,
                 no_iid_median_info=median_age))
        scenario_data.append(
            Node(node_1_data, denorm_node_1_data, col_dist, dataset_name, scenario, config, 'node_1', age_col=age_col,
                 no_iid_median_info=median_age))
        scenario_data.append(
            Node(node_2_data, denorm_node_2_data, col_dist, dataset_name, scenario, config, 'node_2', age_col=age_col,
                 no_iid_median_info=median_age))

    return scenario_data


def transform_time(train_df, denorm_train_df, col_dist=None, test_df=None, denorm_test_df=None, to_gaussian=True):
    transf_train = train_df.copy()
    transf_test = None
    transf_col_dist = None

    # Now, normalize column as a gaussian distribution
    if to_gaussian:
        # Transform time data for data generation (from weibull to gaussian). This is done in case +1
        transf_train.iloc[:, -2] = denorm_train_df.iloc[:, -2]

        loc = denorm_train_df.iloc[:, -2].mean()
        scale = denorm_train_df.iloc[:, -2].std()
        transf_train.iloc[:, -2] = (transf_train.iloc[:, -2] - loc) / scale

        # Transform time column to gaussian
        transf_col_dist = col_dist.copy()
        transf_col_dist[-2] = ('gaussian', 2)

        # Now test data if available
        if test_df is not None:
            transf_test = test_df.copy()
            transf_test.iloc[:, -2] = denorm_test_df.iloc[:, -2].copy()
            transf_test.iloc[:, -2] = (transf_test.iloc[:, -2] - loc) / scale
    else:
        # Obtain loc and scale from denorm_train_df
        loc = denorm_train_df.iloc[:, -2].mean()
        scale = denorm_train_df.iloc[:, -2].std()

        # Use these parameters to denormalize time oclumn from train_df
        transf_train.iloc[:, -2] = (transf_train.iloc[:, -2] * scale) + loc

        # Now, normalize column as a weibull distribution
        if 0.0 in transf_train.iloc[:, -2].values:
            transf_train.iloc[:, -2] = transf_train.iloc[:, -2] + 1

    return transf_train, transf_test, transf_col_dist


class Node(object):
    def __init__(self, node_data, denorm_node_data, col_dist, dataset_name, scenario, config, node, age_col='age',
                 no_iid_median_info=None, no_col_drop_test_data=None):
        self.datas = [deepcopy(node_data) for _ in range(config[
                                                             'n_seeds'])]  # We save one dataset per seed: the original is always the same, but when we generate data, each seed will have a different dataset
        self.denormalized_data = denorm_node_data
        self.node = node
        self.col_dist = col_dist
        self.dataset_name = dataset_name
        self.scenario = scenario
        self.config = config
        self.no_iid_median_info = no_iid_median_info
        self.age_col = age_col
        self.no_col_drop_test_data = no_col_drop_test_data  # Only available for node 1 (data with no dropped col)
        # This is used for max_t: Maximum will be the same for each fold, so we can use the first one
        self.tr_time = self.datas[0][0].iloc[:, -2]
        self.te_time = self.datas[0][2].iloc[:, -2]
        # Training parameters
        self.training_params = {'param_comb': config['params'],
                                'batch_size': config['batch_size'],
                                'lr': config['lr'],
                                'device': config['device'],
                                'n_epochs': config['n_epochs']}

        self.model = {}
        self.init_models(config['n_seeds'], config['params'], config['time_dist'])

    def init_models(self, n_seeds, params, time_dist):
        max_t = max([np.amax(self.tr_time), np.amax(self.te_time)])
        for seed in range(n_seeds):
            model_params = {'feat_distributions': self.col_dist[:-2],
                            'latent_dim': params['latent_dim'],
                            'hidden_size': params['hidden_size'],
                            'input_dim': len(self.col_dist[:-2]),
                            'max_t': max_t,
                            'time_dist': time_dist,
                            'early_stop': True}
            self.model[seed] = SAVAE(model_params)

    def run(self, seed, fed_epochs=0, save_best_model=True, no_col_drop_test=False):
        # Manage data
        x_train = self.datas[seed][0]
        x_test = self.datas[seed][2]
        mask_train = self.datas[seed][1]
        mask_test = self.datas[seed][3]

        # Change number of epochs for federated learning
        if fed_epochs > 0:
            self.training_params['n_epochs'] = fed_epochs
        else:
            self.training_params[
                'n_epochs'] = 1000  # high value to use early stopping (some seeds get stuck: their loss improve slowly, but not their C-index, so we set a max of 1000 epochs, which is usually enough)

        # Train model
        if no_col_drop_test:
            no_col_drop_test_df = self.no_col_drop_test_data[0]
            no_col_drop_test_mask = self.no_col_drop_test_data[1]
            training_results = self.model[seed].fit((x_train, mask_train, x_test, mask_test), self.training_params,
                                                    save_best_model, (no_col_drop_test_df, no_col_drop_test_mask),
                                                    no_col_drop_stop=True)
        else:
            training_results = self.model[seed].fit((x_train, mask_train, x_test, mask_test), self.training_params,
                                                    save_best_model)
        return training_results

    def compute_metrics(self, seed):
        # Configure input data and missing data mask
        x_train = self.datas[seed][0]
        x_test = self.datas[seed][2]
        time_train = np.array(x_train.loc[:, 'time'])
        censor_test = np.array(x_test.loc[:, 'event'])

        # Calculate metrics
        ci, ibs = self.model[seed].calculate_risk(time_train, x_test, censor_test)
        return ci, ibs

    def save(self, training_results, output_path, seed):
        self.model[seed].save(output_path + 'model')
        dictionary = {'input_dim': len(self.col_dist[:-2]),
                      'col_dist': self.col_dist[:-2],
                      'latent_dim': self.training_params['param_comb']['latent_dim'],
                      'hidden_size': self.training_params['param_comb']['hidden_size']}
        for key in training_results.keys():
            dictionary[key] = training_results[key]

        with open(output_path + 'training_results.pickle', 'wb') as handle:
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def generate_synthetic_data(self, output_path, technique, seed):
        # Manage data
        train_df = self.datas[seed][0].copy()
        test_df = self.datas[seed][2].copy()
        denorm_train_df = self.denormalized_data[0].copy()
        train_mask = self.datas[seed][1].copy()
        test_mask = self.datas[seed][3].copy()
        transf_train = train_df.copy()
        transf_test = test_df.copy()
        transf_gen_col_dist = self.col_dist.copy()

        # Modify time here: make it Gaussian and then change the list of distribution
        mean, std = transf_train['time'].mean(), transf_train['time'].std()
        transf_train['time'] = (transf_train['time'] - mean) / std
        transf_test['time'] = (transf_test['time'] - mean) / std
        transf_gen_col_dist[-2] = ('gaussian', 2)

        # Train generator and save data
        data = (transf_train, train_mask, transf_test, test_mask)
        generate(data, denorm_train_df, transf_gen_col_dist, output_path, technique, self.config)

        # Load synthetic data
        model_gen_data_path = output_path + 'gen_seed_0' + os.sep
        gen_data = check_file(model_gen_data_path + 'processed_generated_data.csv', 'Synthetic data does not exist.',
                              csv=True)

        tsne = TSNE(n_components=2, random_state=1234).fit_transform(np.vstack([gen_data, train_df, test_df]))
        # Plot the 3 datasets
        plt.plot(tsne[:len(gen_data), 0], tsne[:len(gen_data), 1], 'o', label='Generated data')
        plt.plot(tsne[len(gen_data):len(gen_data) + len(train_df), 0],
                 tsne[len(gen_data):len(gen_data) + len(train_df), 1], 'o', label='Train data')
        plt.plot(tsne[len(gen_data) + len(train_df):, 0], tsne[len(gen_data) + len(train_df):, 1], 'o',
                 label='Test data')
        plt.legend()
        plt.savefig(model_gen_data_path + 'tsne.png')
        plt.close()

        if gen_data['time'].min() < 0:
            print('Negative values in time column in generated data')

        return gen_data

    def concat_shared_data(self, gen_data, seed):  # Do it for each seed
        best_node_gen_data = gen_data.copy()
        tr_data = self.datas[seed][0].copy()
        tr_mask = self.datas[seed][1].copy()
        test_data = self.datas[seed][2].copy()
        test_mask = self.datas[seed][3].copy()

        # Concatenate gen_data to the training data
        gen_mask = best_node_gen_data.isna().replace([True, False], [0, 1])
        concat_data = pd.concat([tr_data, best_node_gen_data]).reset_index(drop=True)
        concat_mask = pd.concat([tr_mask, gen_mask]).reset_index(drop=True)

        # Shuffle concat_data and concat_mask
        concat_data = concat_data.sample(frac=1, random_state=1234).reset_index(drop=True)
        concat_mask = concat_mask.sample(frac=1, random_state=1234).reset_index(drop=True)

        # Rewrite training data in self.datas[seed]. It is a tuple, so we need to create a new one!!
        self.datas[seed] = (concat_data, concat_mask, test_data, test_mask)

    def obtain_shared_biased_data(self, gen_data, seed, n_data):
        # First, we obtain the projection of the data used for training this seed
        real_data = self.datas[seed][0].copy()
        real_latent = self.model[seed].get_latent(real_data.values[:, 0:-2])[
            0]  # NOTE: we compute the distance over the means of the latents only, as the STDs are an average
        gen_latent = self.model[seed].get_latent(gen_data.values[:, 0:-2])[0]

        # Now, obtain the minimum distance from each latent point to all the real data
        gen_dists = np.ones((gen_latent.shape[0], real_latent.shape[0])) * np.inf
        for i in range(gen_latent.shape[0]):  # Note: this may not be the most efficient implementation...
            for j in range(real_latent.shape[0]):
                gen_dists[i, j] = np.linalg.norm(gen_latent[i] - real_latent[j])
        # Now, obtain the minimum distance for each generated point
        min_dists = np.min(gen_dists, axis=1)
        # Now, sort the generated data by the minimum distance
        sorted_indices = np.argsort(min_dists)
        gen_data = gen_data.iloc[sorted_indices[:n_data]].reset_index(drop=True)
        return gen_data  # This data is now biased
