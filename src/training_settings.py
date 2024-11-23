# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 08/10/2024


# Packages to import
import os

import pandas as pd

from copy import deepcopy

from sklearn.neural_network import MLPRegressor


def train_iso_savae(scenario_data, seed, output_path, config):
    # Prepare savae environment
    n_nodes = len(scenario_data)
    param_folder = str(config['params']['latent_dim']) + '_' + str(config['params']['hidden_size'])

    sa_results = {}
    for i in range(n_nodes):
        sa_results['node_' + str(i)] = {}
        model_path = output_path + 'node_' + str(i) + os.sep + param_folder + os.sep + 'seed_' + str(
            seed) + os.sep + 'iso' + os.sep

        # Train model
        training_results = scenario_data[i].run(seed)

        # Save model information
        scenario_data[i].save(training_results, model_path, seed)
        sa_results['node_' + str(i)]['seed'] = seed
        sa_results['node_' + str(i)]['ci'] = training_results['ci_va']
        sa_results['node_' + str(i)]['ibs'] = training_results['ibs_va']
        sa_results['node_' + str(i)]['n_node'] = i
        sa_results['node_' + str(i)]['tr_loss'] = training_results['loss_tr']
        sa_results['node_' + str(i)]['val_loss'] = training_results['loss_va']

    return sa_results


def train_fed_avg_savae(scenario_data, seed, output_path, config):
    # Prepare savae environment
    n_nodes = len(scenario_data)
    param_folder = str(config['params']['latent_dim']) + '_' + str(config['params']['hidden_size'])
    if config['n_epochs'] > 0:
        local_epochs = config['n_epochs'] // config['fed_steps']
    else:
        local_epochs = -1  # This indicates using early stopping
    fed_steps = config['fed_steps']

    training_stats = [[] for _ in range(n_nodes)]
    save_best_model = False
    for f_step in range(fed_steps):
        # If last step, save best model
        if f_step == fed_steps - 1:
            save_best_model = True
        # Train each node locally
        nodes_w = []
        for i in range(n_nodes):
            # Train model
            training_stats[i].append(scenario_data[i].run(seed, local_epochs, save_best_model))
            nodes_w.append(scenario_data[i].datas[seed][0].shape[0])

        # Now, share the weights!
        # Don't want to share in last epoch so that test C-index is computed on trained model after weight sharing
        if f_step != fed_steps - 1:
            weights = []
            for i in range(n_nodes):
                weights.append(scenario_data[i].model[seed].state_dict())  # Add the local models to the list
            new_weights = deepcopy(weights[0])
            for key in new_weights:  # Vanilla FedAvg
                new_weights[key] = sum([nodes_w[i] * w[key] for i, w in enumerate(weights)]) / sum(nodes_w)

            for i in range(n_nodes):
                scenario_data[i].model[seed].load_state_dict(new_weights)

                # Save model information
                fed_path = output_path + 'node_' + str(i) + os.sep + param_folder + os.sep + 'seed_' + str(
                    seed) + os.sep + 'fed_avg' + os.sep
                scenario_data[i].save(training_stats[i][-1], fed_path + 'fed_step_' + str(f_step) + '_',
                                      seed)  # -1 is the last f_step

    # Save final weights
    sa_results = {}
    for i in range(n_nodes):
        sa_results['node_' + str(i)] = {}
        fed_path = output_path + 'node_' + str(i) + os.sep + param_folder + os.sep + 'seed_' + str(
            seed) + os.sep + 'fed_avg' + os.sep
        scenario_data[i].save(training_stats[i][-1], fed_path, seed)

        # Save ci and ibs for every epoch
        epochs_ci = []
        epochs_ibs = []
        tr_loss = []
        va_loss = []
        for j, stats in enumerate(training_stats[i]):
            epochs_ci.extend(stats['ci_va'])
            epochs_ibs.extend(stats['ibs_va'])
            tr_loss.extend(stats['loss_tr'])
            va_loss.extend(stats['loss_va'])

        sa_results['node_' + str(i)]['ci'] = epochs_ci
        sa_results['node_' + str(i)]['ibs'] = epochs_ibs
        sa_results['node_' + str(i)]['tr_loss'] = tr_loss
        sa_results['node_' + str(i)]['val_loss'] = va_loss
        sa_results['node_' + str(i)]['seed'] = seed
        sa_results['node_' + str(i)]['n_node'] = i

    return sa_results


# Train nodes in a federated way sharing synthetic data in first federated step to see benefit. Then, train as usual.
def train_fed_syn_data_savae(scenario_data, seed, output_path, gen_data_path, config, iid_share=True):
    # Prepare savae environment
    n_nodes = len(scenario_data)
    param_folder = str(config['params']['latent_dim']) + '_' + str(config['params']['hidden_size'])
    if config['n_epochs'] > 0:
        local_epochs = config['n_epochs'] // config['fed_steps']
    else:
        local_epochs = -1  # This indicates using early stopping
    fed_steps = 2

    training_stats = [[] for _ in range(n_nodes)]
    for f_step in range(fed_steps):
        # Train each node locally
        save_best_model = False
        for i in range(n_nodes):
            # If fed step != 0, do not train node 1, replicate previous results in training_stats
            if i == 0:
                if f_step == 0:
                    save_best_model = True
                    node_0_tr_stats = scenario_data[i].run(seed, local_epochs, save_best_model)
                    best_model_ci_va = node_0_tr_stats['ci_va'][-1]
                    best_model_ibs_va = node_0_tr_stats['ibs_va'][-1]
                    # Remove the last element of the lists
                    node_0_tr_stats['ci_va'] = node_0_tr_stats['ci_va'][:-1]
                    node_0_tr_stats['ibs_va'] = node_0_tr_stats['ibs_va'][:-1]
                    # Append to training_stats
                    training_stats[i].append(node_0_tr_stats)
                else:
                    node_0_tr_stats = training_stats[i][-1]
                    if f_step == fed_steps - 1:
                        # Add best_model metrics
                        node_0_tr_stats['ci_va'].append(best_model_ci_va)
                        node_0_tr_stats['ibs_va'].append(best_model_ibs_va)
                    training_stats[i].append(node_0_tr_stats)
            # Train model
            else:
                if f_step == fed_steps - 1:
                    save_best_model = True  # Save best model in last step (for test C-index)
                training_stats[i].append(scenario_data[i].run(seed, local_epochs, save_best_model))

        # After training locally, generate and share patients. Only in first federated step and the best node (i=0)!!
        if f_step == 0:
            # Generate synthetic data
            node_output_path = gen_data_path + 'node_0' + os.sep + config[
                'gen_data_technique'] + os.sep + param_folder + os.sep + 'seed_' + str(seed) + os.sep
            gen_data = scenario_data[0].generate_synthetic_data(node_output_path, config['gen_data_technique'], seed)

            # Share data with other nodes
            if iid_share:
                shared_gen_data = gen_data.copy()[:config['shared_n_gen']]
                for i in range(n_nodes):
                    if i != 0:
                        scenario_data[i].concat_shared_data(shared_gen_data, seed)
            else:
                for i in range(n_nodes):
                    if i != 0:
                        shared_gen_data = gen_data.copy()  # Data to be shared, but note that first we must filter it!
                        # Obtain the latent variable of the training data
                        shared_biased_data = scenario_data[i].obtain_shared_biased_data(shared_gen_data, seed,
                                                                                        config['shared_n_gen'])
                        scenario_data[i].concat_shared_data(shared_biased_data, seed)

            # Check if scenario_data[i].denorm_data has below 0 values
            for i in range(n_nodes):
                if scenario_data[i].denormalized_data[0]['time'].min() < 0:
                    print(f'Error: Time column in scenario_data[{i}].denorm_data has below 0 values')

        # Save model information
        if f_step != fed_steps - 1:
            for i in range(n_nodes):
                fed_path = output_path + 'node_' + str(i) + os.sep + param_folder + os.sep + 'seed_' + str(
                    seed) + os.sep + 'fed_syn_' + ('naive' if iid_share else 'bias') + os.sep
                scenario_data[i].save(training_stats[i][-1], fed_path + 'fed_step_' + str(f_step) + '_',
                                      seed)  # -1 is the last f_step

    # Save final weights
    sa_results = {}
    for i in range(n_nodes):
        sa_results['node_' + str(i)] = {}
        fed_path = output_path + 'node_' + str(i) + os.sep + param_folder + os.sep + 'seed_' + str(
            seed) + os.sep + 'fed_syn_' + ('naive' if iid_share else 'bias') + os.sep
        scenario_data[i].save(training_stats[i][-1], fed_path, seed)

        # Save ci and ibs for every epoch
        epochs_ci = []
        epochs_ibs = []
        tr_loss = []
        va_loss = []
        for j, stats in enumerate(training_stats[i]):
            epochs_ci.extend(stats['ci_va'])
            epochs_ibs.extend(stats['ibs_va'])
            tr_loss.extend(stats['loss_tr'])
            va_loss.extend(stats['loss_va'])

        sa_results['node_' + str(i)]['ci'] = epochs_ci
        sa_results['node_' + str(i)]['ibs'] = epochs_ibs
        sa_results['node_' + str(i)]['tr_loss'] = tr_loss
        sa_results['node_' + str(i)]['val_loss'] = va_loss
        sa_results['node_' + str(i)]['seed'] = seed
        sa_results['node_' + str(i)]['n_node'] = i

    return sa_results


def train_fed_imp(scenario_data, seed, output_path, gen_data_path, config):
    # Prepare savae environment
    n_nodes = len(scenario_data)
    param_folder = str(config['params']['latent_dim']) + '_' + str(config['params']['hidden_size'])
    local_epochs = -1  # This indicates using early stopping
    fed_steps = 2
    col = config['pred_cols'][scenario_data[0].dataset_name]

    training_stats = [[] for _ in range(n_nodes)]
    for f_step in range(fed_steps):
        # Train each node locally
        save_best_model = False
        for i in range(n_nodes):
            no_col_drop = True if (
                    f_step != 0 and i == 1) else False  # Use no col drop data to validate once predictions are available

            # If fed step != 0, do not train node 1, replicate previous results in training_stats
            if i == 0:
                if f_step == 0:
                    save_best_model = True
                    node_0_tr_stats = scenario_data[i].run(seed, local_epochs, save_best_model,
                                                           no_col_drop_test=no_col_drop)
                    best_model_ci_va = node_0_tr_stats['ci_va'][-1]
                    best_model_ibs_va = node_0_tr_stats['ibs_va'][-1]
                    # Remove the last element of the lists
                    node_0_tr_stats['ci_va'] = node_0_tr_stats['ci_va'][:-1]
                    node_0_tr_stats['ibs_va'] = node_0_tr_stats['ibs_va'][:-1]
                    # Append to training_stats
                    training_stats[i].append(node_0_tr_stats)
                else:
                    node_0_tr_stats = training_stats[i][-1]
                    if f_step == fed_steps - 1:
                        # Add best_model metrics
                        node_0_tr_stats['ci_va'].append(best_model_ci_va)
                        node_0_tr_stats['ibs_va'].append(best_model_ibs_va)
                    training_stats[i].append(node_0_tr_stats)
            # Train model
            else:
                if f_step == fed_steps - 1:
                    save_best_model = True  # Save best model in last step (for test C-index)
                training_stats[i].append(
                    scenario_data[i].run(seed, local_epochs, save_best_model, no_col_drop_test=no_col_drop))

        # After training locally, generate patients. Only in first federated step and the best node (i=0)!!
        if f_step == 0:
            # Generate synthetic data
            node_output_path = gen_data_path + 'node_0' + os.sep + config[
                'gen_data_technique'] + os.sep + param_folder + os.sep + 'seed_' + str(seed) + os.sep
            gen_data = scenario_data[0].generate_synthetic_data(node_output_path, config['gen_data_technique'], seed)

            # In scenario 7, use gen data to train a predictor to be used in poor node to impute local data
            # Take from gen_data all columns except col
            col_idx = gen_data.columns.get_loc(col)
            X = gen_data.copy().drop([col], axis=1)  # As x use gen_data without col value and as y use col value
            y = gen_data.copy().loc[:, col]
            pred_model = MLPRegressor(random_state=1, max_iter=2000, tol=0.1)
            pred_model.fit(X, y)

            # Use the predictor to predict col values from local data in node 2
            node_2_train_data = scenario_data[1].datas[seed][0].copy()
            node_2_train_mask = scenario_data[1].datas[seed][1].copy()
            node_2_train_pred_y = pred_model.predict(node_2_train_data)
            node_2_test_data = scenario_data[1].datas[seed][2].copy()
            node_2_test_mask = scenario_data[1].datas[seed][3].copy()
            node_2_test_pred_y = pred_model.predict(node_2_test_data)

            # Rewrite node 2 data
            # introduce node_2_train_pred_y in node_2_train_data in col_idx position
            new_node_2_train_data = node_2_train_data.copy().iloc[:, :col_idx]
            new_node_2_train_mask = node_2_train_mask.copy().iloc[:, :col_idx]
            new_node_2_test_data = node_2_test_data.copy().iloc[:, :col_idx]
            new_node_2_test_mask = node_2_test_mask.copy().iloc[:, :col_idx]
            # Add the predicted values
            new_node_2_train_data[col] = node_2_train_pred_y
            new_node_2_train_mask[col] = 1
            new_node_2_test_data[col] = node_2_test_pred_y
            new_node_2_test_mask[col] = 1
            # Add the rest of the columns
            new_node_2_train_data = pd.concat([new_node_2_train_data, node_2_train_data.iloc[:, col_idx:]], axis=1)
            new_node_2_train_mask = pd.concat([new_node_2_train_mask, node_2_train_mask.iloc[:, col_idx:]], axis=1)
            new_node_2_test_data = pd.concat([new_node_2_test_data, node_2_test_data.iloc[:, col_idx:]], axis=1)
            new_node_2_test_mask = pd.concat([new_node_2_test_mask, node_2_test_mask.iloc[:, col_idx:]], axis=1)

            # Update scenario_data
            scenario_data[1].datas[seed] = (
                new_node_2_train_data, new_node_2_train_mask, new_node_2_test_data, new_node_2_test_mask)

            # Reinitialize the model (new column)
            new_col_dist = scenario_data[0].col_dist.copy()
            scenario_data[1].col_dist = new_col_dist
            scenario_data[1].init_models(config['n_seeds'], config['params'], config['time_dist'])

            # Check if scenario_data[i].denorm_data has below 0 values
            for i in range(n_nodes):
                if scenario_data[i].denormalized_data[0]['time'].min() < 0:
                    print(f'Error: Time column in scenario_data[{i}].denorm_data has below 0 values')

        # Save model information
        if f_step != fed_steps - 1:
            for i in range(n_nodes):
                fed_path = output_path + 'node_' + str(i) + os.sep + param_folder + os.sep + 'seed_' + str(
                    seed) + os.sep + 'fed_imp' + os.sep
                scenario_data[i].save(training_stats[i][-1], fed_path + 'fed_step_' + str(f_step) + '_',
                                      seed)  # -1 is the last f_step

    # Save final weights
    sa_results = {}
    for i in range(n_nodes):
        sa_results['node_' + str(i)] = {}
        fed_path = output_path + 'node_' + str(i) + os.sep + param_folder + os.sep + 'seed_' + str(
            seed) + os.sep + 'fed_imp' + os.sep
        scenario_data[i].save(training_stats[i][-1], fed_path, seed)

        # Save ci and ibs for every epoch
        epochs_ci = []
        epochs_ibs = []
        epochs_no_col_drop_ci = []
        epochs_no_col_drop_ibs = []
        tr_loss = []
        va_loss = []
        for j, stats in enumerate(training_stats[i]):
            epochs_ci.extend(stats['ci_va'])
            epochs_ibs.extend(stats['ibs_va'])
            epochs_no_col_drop_ci.extend(stats['no_col_drop_ci_va'])
            epochs_no_col_drop_ibs.extend(stats['no_col_drop_ibs_va'])
            tr_loss.extend(stats['loss_tr'])
            va_loss.extend(stats['loss_va'])

        sa_results['node_' + str(i)]['ci'] = epochs_ci
        sa_results['node_' + str(i)]['ibs'] = epochs_ibs
        sa_results['node_' + str(i)]['no_col_drop_ci'] = epochs_no_col_drop_ci
        sa_results['node_' + str(i)]['no_col_drop_ibs'] = epochs_no_col_drop_ibs
        sa_results['node_' + str(i)]['tr_loss'] = tr_loss
        sa_results['node_' + str(i)]['val_loss'] = va_loss
        sa_results['node_' + str(i)]['seed'] = seed
        sa_results['node_' + str(i)]['n_node'] = i

    return sa_results


def train_fed_imp_syn_naive(scenario_data, seed, output_path, gen_data_path, config):
    # Prepare savae environment
    n_nodes = len(scenario_data)
    param_folder = str(config['params']['latent_dim']) + '_' + str(config['params']['hidden_size'])
    local_epochs = -1  # This indicates using early stopping
    fed_steps = 2
    col = config['pred_cols'][scenario_data[0].dataset_name]

    training_stats = [[] for _ in range(n_nodes)]
    for f_step in range(fed_steps):
        # Train each node locally
        save_best_model = False
        for i in range(n_nodes):
            no_col_drop = True if (
                    f_step != 0 and i == 1) else False  # Use no col drop data to validate once predictions are available
            # If fed step != 0, do not train node 1, replicate previous results in training_stats
            if i == 0:
                if f_step == 0:
                    save_best_model = True
                    node_0_tr_stats = scenario_data[i].run(seed, local_epochs, save_best_model,
                                                           no_col_drop_test=no_col_drop)
                    best_model_ci_va = node_0_tr_stats['ci_va'][-1]
                    best_model_ibs_va = node_0_tr_stats['ibs_va'][-1]
                    # Remove the last element of the lists
                    node_0_tr_stats['ci_va'] = node_0_tr_stats['ci_va'][:-1]
                    node_0_tr_stats['ibs_va'] = node_0_tr_stats['ibs_va'][:-1]
                    # Append to training_stats
                    training_stats[i].append(node_0_tr_stats)
                else:
                    node_0_tr_stats = training_stats[i][-1]
                    if f_step == fed_steps - 1:
                        # Add best_model metrics
                        node_0_tr_stats['ci_va'].append(best_model_ci_va)
                        node_0_tr_stats['ibs_va'].append(best_model_ibs_va)
                    training_stats[i].append(node_0_tr_stats)
            # Train model
            else:
                if f_step == fed_steps - 1:
                    save_best_model = True  # Save best model in last step (for test C-index)
                training_stats[i].append(
                    scenario_data[i].run(seed, local_epochs, save_best_model, no_col_drop_test=no_col_drop))

        # After training locally, generate patients. Only in first federated step and the best node (i=0)!!
        if f_step == 0:
            # Generate synthetic data
            node_output_path = gen_data_path + 'node_0' + os.sep + config[
                'gen_data_technique'] + os.sep + param_folder + os.sep + 'seed_' + str(seed) + os.sep
            gen_data = scenario_data[0].generate_synthetic_data(node_output_path, config['gen_data_technique'], seed)

            # In scenario 7, use gen data to train a predictor to be used in poor node to impute local data
            # Take from gen_data all columns except col
            col_idx = gen_data.columns.get_loc(col)
            X = gen_data.copy().drop([col], axis=1)  # As x use gen_data without col value and as y use col value
            y = gen_data.copy().loc[:, col]
            pred_model = MLPRegressor(random_state=1, max_iter=2000, tol=0.1)
            pred_model.fit(X, y)

            # Use the predictor to predict col values from local data in node 2
            node_2_train_data = scenario_data[1].datas[seed][0].copy()
            node_2_train_mask = scenario_data[1].datas[seed][1].copy()
            node_2_train_pred_y = pred_model.predict(node_2_train_data)
            node_2_test_data = scenario_data[1].datas[seed][2].copy()
            node_2_test_mask = scenario_data[1].datas[seed][3].copy()
            node_2_test_pred_y = pred_model.predict(node_2_test_data)

            # Rewrite node 2 data
            # introduce node_2_train_pred_y in node_2_train_data in col_idx position
            new_node_2_train_data = node_2_train_data.copy().iloc[:, :col_idx]
            new_node_2_train_mask = node_2_train_mask.copy().iloc[:, :col_idx]
            new_node_2_test_data = node_2_test_data.copy().iloc[:, :col_idx]
            new_node_2_test_mask = node_2_test_mask.copy().iloc[:, :col_idx]
            # Add the predicted values
            new_node_2_train_data[col] = node_2_train_pred_y
            new_node_2_train_mask[col] = 1
            new_node_2_test_data[col] = node_2_test_pred_y
            new_node_2_test_mask[col] = 1
            # Add the rest of the columns
            new_node_2_train_data = pd.concat([new_node_2_train_data, node_2_train_data.iloc[:, col_idx:]], axis=1)
            new_node_2_train_mask = pd.concat([new_node_2_train_mask, node_2_train_mask.iloc[:, col_idx:]], axis=1)
            new_node_2_test_data = pd.concat([new_node_2_test_data, node_2_test_data.iloc[:, col_idx:]], axis=1)
            new_node_2_test_mask = pd.concat([new_node_2_test_mask, node_2_test_mask.iloc[:, col_idx:]], axis=1)

            # Update scenario_data by adding the predicted values and the synthetic data
            scenario_data[1].datas[seed] = (
                new_node_2_train_data, new_node_2_train_mask, new_node_2_test_data, new_node_2_test_mask)
            shared_gen_data = gen_data.copy()[:config['shared_n_gen']]
            for i in range(n_nodes):
                if i != 0:
                    scenario_data[i].concat_shared_data(shared_gen_data, seed)

            # Reinitialize the model (new column)
            new_col_dist = scenario_data[0].col_dist.copy()
            scenario_data[1].col_dist = new_col_dist
            scenario_data[1].init_models(config['n_seeds'], config['params'], config['time_dist'])

            # Check if scenario_data[i].denorm_data has below 0 values
            for i in range(n_nodes):
                if scenario_data[i].denormalized_data[0]['time'].min() < 0:
                    print(f'Error: Time column in scenario_data[{i}].denorm_data has below 0 values')

        # Save model information
        if f_step != fed_steps - 1:
            for i in range(n_nodes):
                fed_path = output_path + 'node_' + str(i) + os.sep + param_folder + os.sep + 'seed_' + str(
                    seed) + os.sep + 'fed_imp' + os.sep
                scenario_data[i].save(training_stats[i][-1], fed_path + 'fed_step_' + str(f_step) + '_',
                                      seed)  # -1 is the last f_step

    # Save final weights
    sa_results = {}
    for i in range(n_nodes):
        sa_results['node_' + str(i)] = {}
        fed_path = output_path + 'node_' + str(i) + os.sep + param_folder + os.sep + 'seed_' + str(
            seed) + os.sep + 'fed_imp' + os.sep
        scenario_data[i].save(training_stats[i][-1], fed_path, seed)

        # Save ci and ibs for every epoch
        epochs_ci = []
        epochs_ibs = []
        epochs_no_col_drop_ci = []
        epochs_no_col_drop_ibs = []
        tr_loss = []
        va_loss = []
        for j, stats in enumerate(training_stats[i]):
            epochs_ci.extend(stats['ci_va'])
            epochs_ibs.extend(stats['ibs_va'])
            epochs_no_col_drop_ci.extend(stats['no_col_drop_ci_va'])
            epochs_no_col_drop_ibs.extend(stats['no_col_drop_ibs_va'])
            tr_loss.extend(stats['loss_tr'])
            va_loss.extend(stats['loss_va'])

        sa_results['node_' + str(i)]['ci'] = epochs_ci
        sa_results['node_' + str(i)]['ibs'] = epochs_ibs
        sa_results['node_' + str(i)]['no_col_drop_ci'] = epochs_no_col_drop_ci
        sa_results['node_' + str(i)]['no_col_drop_ibs'] = epochs_no_col_drop_ibs
        sa_results['node_' + str(i)]['tr_loss'] = tr_loss
        sa_results['node_' + str(i)]['val_loss'] = va_loss
        sa_results['node_' + str(i)]['seed'] = seed
        sa_results['node_' + str(i)]['n_node'] = i

    return sa_results


def train_fed_imp_syn_bias(scenario_data, seed, output_path, gen_data_path, config):
    # Prepare savae environment
    n_nodes = len(scenario_data)
    param_folder = str(config['params']['latent_dim']) + '_' + str(config['params']['hidden_size'])
    local_epochs = -1  # This indicates using early stopping
    fed_steps = 2
    col = config['pred_cols'][scenario_data[0].dataset_name]

    training_stats = [[] for _ in range(n_nodes)]
    for f_step in range(fed_steps):
        # Train each node locally
        save_best_model = False
        for i in range(n_nodes):
            no_col_drop = True if (
                    f_step != 0 and i == 1) else False  # Use no col drop data to validate once predictions are available

            # If fed step != 0, do not train node 1, replicate previous results in training_stats
            if i == 0:
                if f_step == 0:
                    save_best_model = True
                    node_0_tr_stats = scenario_data[i].run(seed, local_epochs, save_best_model,
                                                           no_col_drop_test=no_col_drop)
                    best_model_ci_va = node_0_tr_stats['ci_va'][-1]
                    best_model_ibs_va = node_0_tr_stats['ibs_va'][-1]
                    # Remove the last element of the lists
                    node_0_tr_stats['ci_va'] = node_0_tr_stats['ci_va'][:-1]
                    node_0_tr_stats['ibs_va'] = node_0_tr_stats['ibs_va'][:-1]
                    # Append to training_stats
                    training_stats[i].append(node_0_tr_stats)
                else:
                    node_0_tr_stats = training_stats[i][-1]
                    if f_step == fed_steps - 1:
                        # Add best_model metrics
                        node_0_tr_stats['ci_va'].append(best_model_ci_va)
                        node_0_tr_stats['ibs_va'].append(best_model_ibs_va)
                    training_stats[i].append(node_0_tr_stats)
            # Train model
            else:
                if f_step == fed_steps - 1:
                    save_best_model = True  # Save best model in last step (for test C-index)
                training_stats[i].append(
                    scenario_data[i].run(seed, local_epochs, save_best_model, no_col_drop_test=no_col_drop))

        # After training locally, generate patients. Only in first federated step and the best node (i=0)!!
        if f_step == 0:
            # Generate synthetic data
            node_output_path = gen_data_path + 'node_0' + os.sep + config[
                'gen_data_technique'] + os.sep + param_folder + os.sep + 'seed_' + str(seed) + os.sep
            gen_data = scenario_data[0].generate_synthetic_data(node_output_path, config['gen_data_technique'], seed)

            # In scenario 7, use gen data to train a predictor to be used in poor node to impute local data
            # Take from gen_data all columns except col
            col_idx = gen_data.columns.get_loc(col)
            X = gen_data.copy().drop([col], axis=1)  # As x use gen_data without col value and as y use col value
            y = gen_data.copy().loc[:, col]
            pred_model = MLPRegressor(random_state=1, max_iter=2000, tol=0.1)
            pred_model.fit(X, y)

            # Use the predictor to predict col values from local data in node 2
            node_2_train_data = scenario_data[1].datas[seed][0].copy()
            node_2_train_mask = scenario_data[1].datas[seed][1].copy()
            node_2_train_pred_y = pred_model.predict(node_2_train_data)
            node_2_test_data = scenario_data[1].datas[seed][2].copy()
            node_2_test_mask = scenario_data[1].datas[seed][3].copy()
            node_2_test_pred_y = pred_model.predict(node_2_test_data)

            # Rewrite node 2 data
            # introduce node_2_train_pred_y in node_2_train_data in col_idx position
            new_node_2_train_data = node_2_train_data.copy().iloc[:, :col_idx]
            new_node_2_train_mask = node_2_train_mask.copy().iloc[:, :col_idx]
            new_node_2_test_data = node_2_test_data.copy().iloc[:, :col_idx]
            new_node_2_test_mask = node_2_test_mask.copy().iloc[:, :col_idx]
            # Add the predicted values
            new_node_2_train_data[col] = node_2_train_pred_y
            new_node_2_train_mask[col] = 1
            new_node_2_test_data[col] = node_2_test_pred_y
            new_node_2_test_mask[col] = 1
            # Add the rest of the columns
            new_node_2_train_data = pd.concat([new_node_2_train_data, node_2_train_data.iloc[:, col_idx:]], axis=1)
            new_node_2_train_mask = pd.concat([new_node_2_train_mask, node_2_train_mask.iloc[:, col_idx:]], axis=1)
            new_node_2_test_data = pd.concat([new_node_2_test_data, node_2_test_data.iloc[:, col_idx:]], axis=1)
            new_node_2_test_mask = pd.concat([new_node_2_test_mask, node_2_test_mask.iloc[:, col_idx:]], axis=1)

            # Update scenario_data by adding the predicted values and the synthetic data
            scenario_data[1].datas[seed] = (
                new_node_2_train_data, new_node_2_train_mask, new_node_2_test_data, new_node_2_test_mask)
            # Reinitialize the model (new column)
            new_col_dist = scenario_data[0].col_dist.copy()
            scenario_data[1].col_dist = new_col_dist
            scenario_data[1].init_models(config['n_seeds'], config['params'], config['time_dist'])
            # Concatenate synthetic data
            for i in range(n_nodes):
                if i != 0:
                    shared_gen_data = gen_data.copy()  # Data to be shared, but note that first we must filter it!
                    # Obtain the latent variable of the training data
                    shared_biased_data = scenario_data[i].obtain_shared_biased_data(shared_gen_data, seed,
                                                                                    config['shared_n_gen'])
                    scenario_data[i].concat_shared_data(shared_biased_data, seed)

            # Check if scenario_data[i].denorm_data has below 0 values
            for i in range(n_nodes):
                if scenario_data[i].denormalized_data[0]['time'].min() < 0:
                    print(f'Error: Time column in scenario_data[{i}].denorm_data has below 0 values')

        # Save model information
        if f_step != fed_steps - 1:
            for i in range(n_nodes):
                fed_path = output_path + 'node_' + str(i) + os.sep + param_folder + os.sep + 'seed_' + str(
                    seed) + os.sep + 'fed_imp' + os.sep
                scenario_data[i].save(training_stats[i][-1], fed_path + 'fed_step_' + str(f_step) + '_',
                                      seed)  # -1 is the last f_step

    # Save final weights
    sa_results = {}
    for i in range(n_nodes):
        sa_results['node_' + str(i)] = {}
        fed_path = output_path + 'node_' + str(i) + os.sep + param_folder + os.sep + 'seed_' + str(
            seed) + os.sep + 'fed_imp' + os.sep
        scenario_data[i].save(training_stats[i][-1], fed_path, seed)

        # Save ci and ibs for every epoch
        epochs_ci = []
        epochs_ibs = []
        epochs_no_col_drop_ci = []
        epochs_no_col_drop_ibs = []
        tr_loss = []
        va_loss = []
        for j, stats in enumerate(training_stats[i]):
            epochs_ci.extend(stats['ci_va'])
            epochs_ibs.extend(stats['ibs_va'])
            epochs_no_col_drop_ci.extend(stats['no_col_drop_ci_va'])
            epochs_no_col_drop_ibs.extend(stats['no_col_drop_ibs_va'])
            tr_loss.extend(stats['loss_tr'])
            va_loss.extend(stats['loss_va'])

        sa_results['node_' + str(i)]['ci'] = epochs_ci
        sa_results['node_' + str(i)]['ibs'] = epochs_ibs
        sa_results['node_' + str(i)]['no_col_drop_ci'] = epochs_no_col_drop_ci
        sa_results['node_' + str(i)]['no_col_drop_ibs'] = epochs_no_col_drop_ibs
        sa_results['node_' + str(i)]['tr_loss'] = tr_loss
        sa_results['node_' + str(i)]['val_loss'] = va_loss
        sa_results['node_' + str(i)]['seed'] = seed
        sa_results['node_' + str(i)]['n_node'] = i

    return sa_results
