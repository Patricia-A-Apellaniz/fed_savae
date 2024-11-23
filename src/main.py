# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 08/10/2024


# Packages to import
import os
import sys
import pickle

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from tabulate import tabulate
from src.node import manage_data
from joblib import delayed, Parallel
from scipy.stats import ttest_ind_from_stats
from results_display import show_best_results
from statsmodels.stats.multitest import multipletests
from utils import create_output_dir, save, check_file, run_config, fill_sequence
from src.training_settings import train_iso_savae, train_fed_avg_savae, train_fed_syn_data_savae


def main():
    print('\n\n-------- FEDERATED SURVIVAL ANALYSIS  --------')

    # Environment configuration
    task = 'survival_analysis'
    config = run_config()
    create_output_dir(config, task)
    iso = config['iso']
    fed_avg = config['fed_avg']
    fed_syn_naive = config['fed_syn_naive']
    fed_syn_bias = config['fed_syn_bias']

    # Preprocess data
    if config['train']:
        for dataset_name in config['datasets']:
            print('\n' + '-' * 50)
            print(f'\n{dataset_name} dataset')

            # Load dataset
            # Input data is already processed! PROBLEM: We need to denormalize time column (it has been treated as a gaussian) and then treat it as a weibull distribution
            df = pd.read_csv(config['input_path'] + dataset_name + os.sep + 'data.csv')
            # Therefore we need the raw data to denormalize the time column, but just the training data with which the generator was trained  --> NO! Data was processed entirely!
            original_df = pd.read_csv(config['input_path'] + dataset_name + os.sep + 'raw_data.csv')

            for scenario in config['scenarios']:

                # Configure the scenario data
                scenario_data = manage_data(df, original_df, scenario, dataset_name, config, col_dist=None)


                # Train savae model
                output_path = config['output_path'] + dataset_name + os.sep + scenario + os.sep
                print(f'\nRunning {scenario}...')
                # Save results in dictionary
                params = str(config['params']['latent_dim']) + '_' + str(config['params']['hidden_size'])
                if os.path.exists(output_path + 'results_information.pickle') and config['n_epochs'] > 0:
                    # Use already saved results dict so other training setting does not get removed
                    file = open(output_path + 'results_information.pickle', 'rb')
                    results = pickle.load(file)
                    file.close()
                else:
                    results = {'iso': {}, 'fed_avg': {}, 'fed_syn_naive': {}, 'fed_syn_bias': {}}
                    for i in range(len(scenario_data)):
                        if config['n_epochs'] > 0:
                            results['iso']['node_' + str(i)] = {
                                params: {'ci': np.zeros((config['n_seeds'], config['n_epochs'] + 1, 3)),
                                         'ibs': np.zeros((config['n_seeds'], config['n_epochs'] + 1, 3))}}
                            results['fed_avg']['node_' + str(i)] = {
                                params: {'ci': np.zeros((config['n_seeds'], config['n_epochs'] + 1, 3)),
                                         'ibs': np.zeros((config['n_seeds'], config['n_epochs'] + 1, 3))}}
                            results['fed_syn_naive']['node_' + str(i)] = {
                                params: {'ci': np.zeros((config['n_seeds'], config['n_epochs'] + 1, 3)),
                                         'ibs': np.zeros((config['n_seeds'], config['n_epochs'] + 1, 3))}}
                            results['fed_syn_bias']['node_' + str(i)] = {
                                params: {'ci': np.zeros((config['n_seeds'], config['n_epochs'] + 1, 3)),
                                         'ibs': np.zeros((config['n_seeds'], config['n_epochs'] + 1, 3))}}

                # Train isolated models
                if iso:
                    iso_tr_results = Parallel(n_jobs=config['n_threads'], verbose=10)(delayed(train_iso_savae)(scenario_data, seed, output_path, config) for seed in range(config['n_seeds']))

                    if config['n_epochs'] > 0:
                        for res in iso_tr_results:
                            for i in range(len(scenario_data)):
                                val = res['node_' + str(i)]
                                seed = val['seed']
                                results['iso']['node_' + str(i)][params]['ci'][seed, :] = np.array(val['ci'])
                                results['iso']['node_' + str(i)][params]['ibs'][seed, :] = np.array(val['ibs'])
                    else:
                        # Find the maximum number of epochs
                        max_epochs = 0
                        for res in iso_tr_results:
                            for i in range(len(scenario_data)):
                                val = res['node_' + str(i)]
                                max_epochs = max(max_epochs, len(val['ci']))
                        # Append the results to the dictionary
                        for i in range(len(scenario_data)):
                            results['iso']['node_' + str(i)] = {
                                params: {'ci': np.zeros((config['n_seeds'], max_epochs + 1, 3)),
                                         'ibs': np.zeros((config['n_seeds'], max_epochs + 1, 3))}}
                        for res in iso_tr_results:
                            for i in range(len(scenario_data)):
                                val = res['node_' + str(i)]
                                seed = val['seed']
                                results['iso']['node_' + str(i)][params]['ci'][seed, :] = fill_sequence(np.array(val['ci']), max_epochs + 1)
                                results['iso']['node_' + str(i)][params]['ibs'][seed, :] = fill_sequence(np.array(val['ibs']), max_epochs + 1)

                # If scenario is not centralized, train federated models
                if scenario != 'centralized':
                    # Train federated average models
                    if fed_avg:
                        fed_avg_tr_results = Parallel(n_jobs=config['n_threads'], verbose=10)(
                            delayed(train_fed_avg_savae)(scenario_data, seed, output_path, config) for seed
                            in
                            range(config['n_seeds']))

                        if config['n_epochs'] > 0:
                            for res in fed_avg_tr_results:
                                for i in range(len(scenario_data)):
                                    val = res['node_' + str(i)]
                                    seed = val['seed']
                                    results['fed_avg']['node_' + str(i)][params]['ci'][seed, :] = np.array(val['ci'])
                                    results['fed_avg']['node_' + str(i)][params]['ibs'][seed, :] = np.array(val['ibs'])
                        else:
                            # Find the maximum number of epochs
                            max_epochs = 0
                            for res in fed_avg_tr_results:
                                for i in range(len(scenario_data)):
                                    val = res['node_' + str(i)]
                                    max_epochs = max(max_epochs, len(val['ci']))
                            # Append the results to the dictionary
                            for i in range(len(scenario_data)):
                                results['fed_avg']['node_' + str(i)] = {
                                    params: {'ci': np.zeros((config['n_seeds'], max_epochs + 1, 3)),
                                             'ibs': np.zeros((config['n_seeds'], max_epochs + 1, 3))}}
                            for res in fed_avg_tr_results:
                                for i in range(len(scenario_data)):
                                    val = res['node_' + str(i)]
                                    seed = val['seed']
                                    results['fed_avg']['node_' + str(i)][params]['ci'][seed, :] = fill_sequence(np.array(val['ci']), max_epochs + 1)
                                    results['fed_avg']['node_' + str(i)][params]['ibs'][seed, :] = fill_sequence(np.array(val['ibs']), max_epochs + 1)

                    # Train federated models with synthetic data
                    if fed_syn_naive:
                        gen_path = config['gen_data_path'] + dataset_name + os.sep + scenario + os.sep + 'naive' + os.sep
                        # Create output directory
                        create_output_dir(config, 'fed_syn_data_gen')
                        fed_syn_tr_naive_results = Parallel(n_jobs=config['n_threads'], verbose=10)(
                            delayed(train_fed_syn_data_savae)(scenario_data, seed, output_path, gen_path,
                                                              config, iid_share=True) for seed in range(config['n_seeds']))
                        if config['n_epochs'] > 0:
                            for res in fed_syn_tr_naive_results:
                                for i in range(len(scenario_data)):
                                    val = res['node_' + str(i)]
                                    seed = val['seed']
                                    results['fed_syn_naive']['node_' + str(i)][params]['ci'][seed, :] = np.array(val['ci'])
                                    results['fed_syn_naive']['node_' + str(i)][params]['ibs'][seed, :] = np.array(val['ibs'])
                        else:
                            # Find the maximum number of epochs
                            max_epochs = 0
                            for res in fed_syn_tr_naive_results:
                                for i in range(len(scenario_data)):
                                    val = res['node_' + str(i)]
                                    max_epochs = max(max_epochs, len(val['ci']))
                            # Append the results to the dictionary
                            for i in range(len(scenario_data)):
                                results['fed_syn_naive']['node_' + str(i)] = {
                                    params: {'ci': np.zeros((config['n_seeds'], max_epochs + 1, 3)),
                                             'ibs': np.zeros((config['n_seeds'], max_epochs + 1, 3))}}
                            for res in fed_syn_tr_naive_results:
                                for i in range(len(scenario_data)):
                                    val = res['node_' + str(i)]
                                    seed = val['seed']
                                    results['fed_syn_naive']['node_' + str(i)][params]['ci'][seed, :] = fill_sequence(np.array(val['ci']), max_epochs + 1)
                                    results['fed_syn_naive']['node_' + str(i)][params]['ibs'][seed, :] = fill_sequence(np.array(val['ibs']), max_epochs + 1)

                    if fed_syn_bias:
                        gen_path = config['gen_data_path'] + dataset_name + os.sep + scenario + os.sep + 'bias' + os.sep
                        # Create output directory
                        create_output_dir(config, 'fed_syn_data_gen')
                        fed_syn_tr_bias_results = Parallel(n_jobs=config['n_threads'], verbose=10)(
                            delayed(train_fed_syn_data_savae)(scenario_data, seed, output_path, gen_path,
                                                              config, iid_share=False) for seed in range(config['n_seeds']))
                        if config['n_epochs'] > 0:
                            for res in fed_syn_tr_bias_results:
                                for i in range(len(scenario_data)):
                                    val = res['node_' + str(i)]
                                    seed = val['seed']
                                    results['fed_syn_bias']['node_' + str(i)][params]['ci'][seed, :] = np.array(val['ci'])
                                    results['fed_syn_bias']['node_' + str(i)][params]['ibs'][seed, :] = np.array(val['ibs'])
                        else:
                            # Find the maximum number of epochs
                            max_epochs = 0
                            for res in fed_syn_tr_bias_results:
                                for i in range(len(scenario_data)):
                                    val = res['node_' + str(i)]
                                    max_epochs = max(max_epochs, len(val['ci']))
                            # Append the results to the dictionary
                            for i in range(len(scenario_data)):
                                results['fed_syn_bias']['node_' + str(i)] = {
                                    params: {'ci': np.zeros((config['n_seeds'], max_epochs + 1, 3)),
                                             'ibs': np.zeros((config['n_seeds'], max_epochs + 1, 3))}}
                            for res in fed_syn_tr_bias_results:
                                for i in range(len(scenario_data)):
                                    val = res['node_' + str(i)]
                                    seed = val['seed']
                                    results['fed_syn_bias']['node_' + str(i)][params]['ci'][seed, :] = fill_sequence(np.array(val['ci']), max_epochs + 1)
                                    results['fed_syn_bias']['node_' + str(i)][params]['ibs'][seed, :] = fill_sequence(np.array(val['ibs']), max_epochs + 1)

                # Save results
                print('Saving results...')
                save(results, output_path + 'results_information.pickle')

    seed_eval = config['seed_eval']
    pv_th = config['pv_th']
    if config['show_results']:
        for dataset_name in config['datasets']:
            uncorrected_ci_p_values_iid = []
            uncorrected_ci_p_values_no_iid = []
            uncorrected_ibs_p_values_iid = []
            uncorrected_ibs_p_values_no_iid = []

            print('\n' + '-' * 50)
            print(f'\n{dataset_name} dataset')
            # Load dataset
            # Input data is already processed! PROBLEM: We need to denormalize time column (it has been treated as a gaussian) and then treat it as a weibull distribution
            df = pd.read_csv(config['input_path'] + dataset_name + os.sep + 'data.csv')
            # Therefore we need the raw data to denormalize the time column, but just the training data with which the generator was trained  --> NO! Data was processed entirely!
            original_df = pd.read_csv(config['input_path'] + dataset_name + os.sep + 'raw_data.csv')

            tab = []
            tab_ibs = []
            for scenario in config['scenarios']:
                # Configure the scenario data
                scenario_data = manage_data(df, original_df, scenario, dataset_name, config, col_dist=None)

                # Train savae model
                output_path = config['output_path'] + dataset_name + os.sep + scenario + os.sep

                # Show results
                results = check_file(output_path + 'results_information.pickle', 'Results file does not exist.')
                #best_results = show_best_results(results, output_path, scenario, config, dataset_name)
                modes = ['iso', 'fed_avg', 'fed_syn_naive', 'fed_syn_bias'] if scenario != 'centralized' else ['iso']
                best_results = show_best_results(results, output_path, scenario, config, dataset_name, modes, seeds_eval=seed_eval)

                # Save information for table
                context = ['iso', 'fed_avg', 'fed_syn_naive', 'fed_syn_bias']
                for node in range(len(scenario_data)):
                    if scenario == 'centralized':
                        t = [scenario, 'Node ' + str(node)]
                        t_ibs = [scenario, 'Node ' + str(node)]
                    else:
                        t = [scenario if node == 1 else '', 'Node ' + str(node)]
                        t_ibs = [scenario if node == 1 else '', 'Node ' + str(node)]
                    for c in context:
                        if scenario == 'centralized' and c != 'iso':
                            t.extend(['-'])
                            t_ibs.extend(['-'])
                        else:
                            ci = str(format(best_results[c]['node_' + str(node)][0], '.3f'))
                            ci_min = str(format(best_results[c]['node_' + str(node)][1], '.3f'))
                            ci_max = str(format(best_results[c]['node_' + str(node)][2], '.3f'))
                            t.extend(['(' + ci_min + ' - ' + ci + ' - ' + ci_max + ')'])

                            ibs = str(format(best_results[c]['node_' + str(node)][4], '.3f'))
                            ibs_min = str(format(best_results[c]['node_' + str(node)][5], '.3f'))
                            ibs_max = str(format(best_results[c]['node_' + str(node)][6], '.3f'))
                            t_ibs.extend(['(' + ibs_min + ' - ' + ibs + ' - ' + ibs_max + ')'])

                    # Obtain p-values for federated models (iso is the reference)
                    if scenario == 'centralized':
                        t.extend(['-', '-'])
                        t_ibs.extend(['-', '-'])
                    else:
                        ci_iso = best_results['iso']['node_' + str(node)][0]
                        seeds_cis_iso = [val[1] for val in best_results['iso']['node_' + str(node)][3]]
                        n_obs_iso = len(seeds_cis_iso)
                        std_iso = np.std(seeds_cis_iso)

                        ci_avg = best_results['fed_avg']['node_' + str(node)][0]
                        seeds_cis_avg = [val[1] for val in best_results['fed_avg']['node_' + str(node)][3]]
                        n_obs_avg = len(seeds_cis_avg)
                        std_avg = np.std(seeds_cis_avg)
                        avg_test_ci = ttest_ind_from_stats(ci_iso, std_iso, n_obs_iso, ci_avg, std_avg, n_obs_avg, equal_var=False, alternative='less')

                        ci_synt_n = best_results['fed_syn_naive']['node_' + str(node)][0]
                        seeds_cis_synt_n = [val[1] for val in best_results['fed_syn_naive']['node_' + str(node)][3]]
                        n_obs_synt_n = len(seeds_cis_synt_n)
                        std_synt_n = np.std(seeds_cis_synt_n)
                        synt_test_ci_n = ttest_ind_from_stats(ci_iso, std_iso, n_obs_iso, ci_synt_n, std_synt_n, n_obs_synt_n, equal_var=False, alternative='less')

                        ci_synt_b = best_results['fed_syn_bias']['node_' + str(node)][0]
                        seeds_cis_synt_b = [val[1] for val in best_results['fed_syn_bias']['node_' + str(node)][3]]
                        n_obs_synt_b = len(seeds_cis_synt_b)
                        std_synt_b = np.std(seeds_cis_synt_b)
                        synt_test_ci_b = ttest_ind_from_stats(ci_iso, std_iso, n_obs_iso, ci_synt_b, std_synt_b, n_obs_synt_b, equal_var=False, alternative='less')

                        ibs_iso = best_results['iso']['node_' + str(node)][4]
                        seeds_ibs_iso = [val[1] for val in best_results['iso']['node_' + str(node)][7]]
                        n_obs_ibs_iso = len(seeds_ibs_iso)  # seed eval
                        std_ibs_iso = np.std(seeds_ibs_iso)

                        ibs_avg = best_results['fed_avg']['node_' + str(node)][4]
                        seeds_ibs_avg = [val[1] for val in best_results['fed_avg']['node_' + str(node)][7]]
                        n_obs_ibs_avg = len(seeds_ibs_avg)
                        std_ibs_avg = np.std(seeds_ibs_avg)
                        avg_test_ibs = ttest_ind_from_stats(ibs_iso, std_ibs_iso, n_obs_ibs_iso, ibs_avg, std_ibs_avg, n_obs_ibs_avg, equal_var=False, alternative='greater')

                        ibs_synt_n = best_results['fed_syn_naive']['node_' + str(node)][4]
                        seeds_ibs_synt_n = [val[1] for val in best_results['fed_syn_naive']['node_' + str(node)][7]]
                        n_obs_ibs_synt_n = len(seeds_ibs_synt_n)
                        std_ibs_synt_n = np.std(seeds_ibs_synt_n)
                        synt_test_ibs_n = ttest_ind_from_stats(ibs_iso, std_ibs_iso, n_obs_ibs_iso, ibs_synt_n, std_ibs_synt_n, n_obs_ibs_synt_n, equal_var=False, alternative='greater')

                        ibs_synt_b = best_results['fed_syn_bias']['node_' + str(node)][4]
                        seeds_ibs_synt_b = [val[1] for val in best_results['fed_syn_bias']['node_' + str(node)][7]]
                        n_obs_ibs_synt_b = len(seeds_ibs_synt_b)
                        std_ibs_synt_b = np.std(seeds_ibs_synt_b)
                        synt_test_ibs_b = ttest_ind_from_stats(ibs_iso, std_ibs_iso, n_obs_ibs_iso, ibs_synt_b, std_ibs_synt_b, n_obs_ibs_synt_b, equal_var=False, alternative='greater')

                        if scenario == 'scenario_2' and node == 1:
                            pass
                        t.extend([str(format(avg_test_ci.pvalue, '.3f')) + ' / ' + str(format(synt_test_ci_n.pvalue, '.3f')) + ' / ' + str(format(synt_test_ci_b.pvalue, '.3f'))])
                        t.extend([('*' if avg_test_ci.pvalue < pv_th else '-') + ' / ' + ('*' if synt_test_ci_n.pvalue < pv_th else '-') + ' / ' + ('*' if synt_test_ci_b.pvalue < pv_th else '-')])

                        t_ibs.extend([str(format(avg_test_ibs.pvalue, '.3f')) + ' / ' + str(format(synt_test_ibs_n.pvalue, '.3f')) + ' / ' + str(format(synt_test_ibs_b.pvalue, '.3f'))])
                        t_ibs.extend([('*' if avg_test_ibs.pvalue < pv_th else '-') + ' / ' + ('*' if synt_test_ibs_n.pvalue < pv_th else '-') + ' / ' + ('*' if synt_test_ibs_b.pvalue < pv_th else '-')])

                        # Add p_values to list to adjust them later
                        if scenario in ['scenario_1', 'scenario_2', 'scenario_3']:
                            uncorrected_ci_p_values_iid.extend(
                                [avg_test_ci.pvalue, synt_test_ci_n.pvalue, synt_test_ci_b.pvalue])
                            uncorrected_ibs_p_values_iid.extend(
                                [avg_test_ibs.pvalue, synt_test_ibs_n.pvalue, synt_test_ibs_b.pvalue])

                        else:
                            uncorrected_ci_p_values_no_iid.extend(
                                [avg_test_ci.pvalue, synt_test_ci_n.pvalue, synt_test_ci_b.pvalue])
                            uncorrected_ibs_p_values_no_iid.extend(
                                [avg_test_ibs.pvalue, synt_test_ibs_n.pvalue, synt_test_ibs_b.pvalue])

                    tab.append(t)
                    tab_ibs.append(t_ibs)

                tab.append(['', '', '', '', '', '', ''])
                tab_ibs.append(['', '', '', '', '', '', ''])

            names = ['Scenario', 'Nodes', 'Isolated', 'Federated Average', 'Federated Synthetic Data', 'Federated Synthetic Data Bias', 'Significant Advantage', 'p_value < ' + str(pv_th) + ' ?']
            print('\n')
            print(tabulate(tab, headers=names, tablefmt='orgtbl'))
            print(tabulate(tab, headers=names, tablefmt='latex'))
            print('\n')
            print(tabulate(tab_ibs, headers=names, tablefmt='orgtbl'))
            print(tabulate(tab_ibs, headers=names, tablefmt='latex'))


            # P VALUE ADJUSTMENT BY DATASET
            data_adjusted_p_values_iid = []
            data_adjusted_p_values_non_iid = []
            # Adjust p-values
            adjusted_ci_p_vals_iid = list(multipletests(uncorrected_ci_p_values_iid, method='holm')[1])
            adjusted_ibs_p_vals_iid = list(multipletests(uncorrected_ibs_p_values_iid, method='holm')[1])
            adjusted_ci_p_vals_no_iid = list(multipletests(uncorrected_ci_p_values_no_iid, method='holm')[1])
            adjusted_ibs_p_vals_no_iid = list(multipletests(uncorrected_ibs_p_values_no_iid, method='holm')[1])

            for scenario in config['scenarios']:
                if scenario == 'centralized':
                    continue
                elif scenario in ['scenario_1', 'scenario_2', 'scenario_3']:
                    for node in range(3):
                        row_ci_adj_p_vals = adjusted_ci_p_vals_iid[:3]
                        del adjusted_ci_p_vals_iid[0]
                        del adjusted_ci_p_vals_iid[0]
                        del adjusted_ci_p_vals_iid[0]
                        if node == 1:
                            t_adj = [scenario, 'Node ' + str(node), str(format(row_ci_adj_p_vals[0], '.3f')) + ' / ' + str(format(row_ci_adj_p_vals[1], '.3f')) + ' / ' + str(format(row_ci_adj_p_vals[2], '.3f'))]
                        else:
                            t_adj = ['', 'Node ' + str(node), str(format(row_ci_adj_p_vals[0], '.3f')) + ' / ' + str(format(row_ci_adj_p_vals[1], '.3f')) + ' / ' + str(format(row_ci_adj_p_vals[2], '.3f'))]
                        t_adj.extend([('*' if row_ci_adj_p_vals[0] < pv_th else '-') + ' / ' + ('*' if row_ci_adj_p_vals[1] < pv_th else '-') + ' / ' + ('*' if row_ci_adj_p_vals[2] < pv_th else '-')])
                        row_ibs_adj_p_vals = adjusted_ibs_p_vals_iid[:3]
                        del adjusted_ibs_p_vals_iid[0]
                        del adjusted_ibs_p_vals_iid[0]
                        del adjusted_ibs_p_vals_iid[0]
                        t_adj.extend([str(format(row_ibs_adj_p_vals[0], '.3f')) + ' / ' + str(format(row_ibs_adj_p_vals[1], '.3f')) + ' / ' + str(format(row_ibs_adj_p_vals[2], '.3f'))])
                        t_adj.extend([('*' if row_ibs_adj_p_vals[0] < pv_th else '-') + ' / ' + ('*' if row_ibs_adj_p_vals[1] < pv_th else '-') + ' / ' + ('*' if row_ibs_adj_p_vals[2] < pv_th else '-')])

                        data_adjusted_p_values_iid.append(t_adj)

                elif scenario in ['scenario_4', 'scenario_5', 'scenario_6']:
                    for node in range(3):
                        row_ci_adj_p_vals = adjusted_ci_p_vals_no_iid[:3]
                        del adjusted_ci_p_vals_no_iid[0]
                        del adjusted_ci_p_vals_no_iid[0]
                        del adjusted_ci_p_vals_no_iid[0]
                        if node == 1:
                            t_adj = [scenario, 'Node ' + str(node),
                                     str(format(row_ci_adj_p_vals[0], '.3f')) + ' / ' + str(
                                         format(row_ci_adj_p_vals[1], '.3f')) + ' / ' + str(
                                         format(row_ci_adj_p_vals[2], '.3f'))]
                        else:
                            t_adj = ['', 'Node ' + str(node), str(format(row_ci_adj_p_vals[0], '.3f')) + ' / ' + str(
                                format(row_ci_adj_p_vals[1], '.3f')) + ' / ' + str(format(row_ci_adj_p_vals[2], '.3f'))]
                        t_adj.extend([('*' if row_ci_adj_p_vals[0] < pv_th else '-') + ' / ' + (
                            '*' if row_ci_adj_p_vals[1] < pv_th else '-') + ' / ' + (
                                          '*' if row_ci_adj_p_vals[2] < pv_th else '-')])
                        row_ibs_adj_p_vals = adjusted_ibs_p_vals_no_iid[:3]
                        del adjusted_ibs_p_vals_no_iid[0]
                        del adjusted_ibs_p_vals_no_iid[0]
                        del adjusted_ibs_p_vals_no_iid[0]
                        t_adj.extend([str(format(row_ibs_adj_p_vals[0], '.3f')) + ' / ' + str(
                            format(row_ibs_adj_p_vals[1], '.3f')) + ' / ' + str(format(row_ibs_adj_p_vals[2], '.3f'))])
                        t_adj.extend([('*' if row_ibs_adj_p_vals[0] < pv_th else '-') + ' / ' + (
                            '*' if row_ibs_adj_p_vals[1] < pv_th else '-') + ' / ' + (
                                          '*' if row_ibs_adj_p_vals[2] < pv_th else '-')])

                        data_adjusted_p_values_non_iid.append(t_adj)

            print('ADJUSTED BY DATASET IID SCENARIOS')
            names = ['Scenario', 'Nodes', 'Adjusted Significant Advantage (CI)', 'adjusted p_value < ' + str(pv_th) + ' ?  (CI)', 'Adjusted Significant Advantage (IBS)', 'adjusted p_value < ' + str(pv_th) + ' ?  (IBS)']
            print(tabulate(data_adjusted_p_values_iid, headers=names, tablefmt='orgtbl'))
            print(tabulate(data_adjusted_p_values_iid, headers=names, tablefmt='latex'))
            print('\n')

            print('ADJUSTED BY DATASET NON-IID SCENARIOS')
            names = ['Scenario', 'Nodes', 'Adjusted Significant Advantage (CI)',
                     'adjusted p_value < ' + str(pv_th) + ' ?  (CI)', 'Adjusted Significant Advantage (IBS)',
                     'adjusted p_value < ' + str(pv_th) + ' ?  (IBS)']
            print(tabulate(data_adjusted_p_values_non_iid, headers=names, tablefmt='orgtbl'))
            print(tabulate(data_adjusted_p_values_non_iid, headers=names, tablefmt='latex'))
            print('\n')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
