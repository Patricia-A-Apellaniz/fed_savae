# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 02/11/2024


# Packages to import
import os
import sys
import pickle

import numpy as np
import pandas as pd

from tabulate import tabulate
from joblib import delayed, Parallel
from scipy.stats import ttest_ind_from_stats
from statsmodels.stats.multitest import multipletests

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from src.node import manage_scenario_7_data
from src.results_display import show_best_results_sc_7
from src.utils import save, check_file, run_config, create_output_dir
from src.training_settings import train_iso_savae, train_fed_imp, train_fed_imp_syn_naive, train_fed_imp_syn_bias


def load_results_dict(scenario_data, output_path, config):
    params = str(config['params']['latent_dim']) + '_' + str(config['params']['hidden_size'])
    modes = ['iso', 'fed_imp', 'fed_imp_syn_naive', 'fed_imp_syn_bias']

    if os.path.exists(output_path + 'results_information.pickle'):
        # Use already saved results dict so other training setting does not get removed
        file = open(output_path + 'results_information.pickle', 'rb')
        results = pickle.load(file)
        file.close()
    else:
        results = {}
        for mode in modes:
            results[mode] = {}
            for i in range(len(scenario_data)):
                results[mode]['node_' + str(i)] = {params: {}}
    return results


def save_results(tr_results, saved_results, mode, scenario_data, config):
    params = str(config['params']['latent_dim']) + '_' + str(config['params']['hidden_size'])

    # Find the maximum number of epochs
    max_epochs = 0
    for res in tr_results:
        for i in range(len(scenario_data)):
            val = res['node_' + str(i)]
            max_epochs = max(max_epochs, len(val['ci']))

    # Append the results to the dictionary
    for i in range(len(scenario_data)):
        saved_results[mode]['node_' + str(i)] = {
            params: {'ci': np.zeros((config['n_seeds'], max_epochs + 1, 3)),
                     'ibs': np.zeros((config['n_seeds'], max_epochs + 1, 3)),
                     'no_col_drop_ci': np.zeros((config['n_seeds'], max_epochs + 1, 3)),
                     'no_col_drop_ibs': np.zeros((config['n_seeds'], max_epochs + 1, 3))}}
    for res in tr_results:
        for i in range(len(scenario_data)):
            val = res['node_' + str(i)]
            seed = val['seed']
            saved_results[mode]['node_' + str(i)][params]['ci'][seed, :] = fill_sequence(np.array(val['ci']),
                                                                                         max_epochs + 1)
            saved_results[mode]['node_' + str(i)][params]['ibs'][seed, :] = fill_sequence(np.array(val['ibs']),
                                                                                          max_epochs + 1)
            if mode != 'iso' and i != 0:
                # Check if the sequence is homogeneous (due to col prediction)
                ci_val = val['no_col_drop_ci']
                ibs_val = val['no_col_drop_ibs']
                # To do so, check if first value is the same dim as last value
                if (ci_val[0] == 0.0) != len(ci_val[-1]):
                    # If not, go throigh whole sequence and check if all of them are tuples of three values. If not, replace the values with (0.0, 0.0, 0.0)
                    for j in range(len(ci_val)):
                        if ci_val[j] == 0.0:
                            ci_val[j] = np.array([0.0, 0.0, 0.0])
                            ibs_val[j] = np.array([0.0, 0.0, 0.0])
                saved_results[mode]['node_' + str(i)][params]['no_col_drop_ci'][seed, :] = fill_sequence(
                    np.array(ci_val), max_epochs + 1)
                saved_results[mode]['node_' + str(i)][params]['no_col_drop_ibs'][seed, :] = fill_sequence(
                    np.array(ibs_val), max_epochs + 1)
    return saved_results


def main():
    print('\n\n-------- FEDERATED SURVIVAL ANALYSIS  --------')

    # Environment configuration
    config = run_config(imp=True)
    create_output_dir(config, task='imp')
    scenario = 'scenario_7'

    # Train models
    if config['train']:
        for dataset in config['datasets']:
            print('\n' + '-' * 50)
            print('\n-------- DATASET: ', dataset, ' --------')

            # Load dataset
            # Input data is already processed! PROBLEM: We need to denormalize time column (it has been treated as a gaussian) and then treat it as a weibull distribution
            df = pd.read_csv(config['input_path'] + dataset + os.sep + 'data.csv')
            # Therefore we need the raw data to denormalize the time column, but just the training data with which the generator was trained  --> NO! Data was processed entirely!
            real_df = pd.read_csv(config['input_path'] + dataset + os.sep + 'raw_data.csv')

            # Configure the scenario data
            col = config['pred_cols'][dataset]
            scenario_data = manage_scenario_7_data(df, real_df, dataset, config)
            output_path = config['output_path'] + dataset + os.sep + scenario + os.sep + col + os.sep

            # Load dictionary to save results
            results = load_results_dict(scenario_data, output_path, config)

            # Train isolated models
            if config['iso']:
                print('\n---- ISOLATED TRAINING  ----')

                iso_results = Parallel(n_jobs=config['n_threads'], verbose=10)(
                    delayed(train_iso_savae)(scenario_data, seed, output_path, config) for seed in
                    range(config['n_seeds']))

                # Save results
                results = save_results(iso_results, results, 'iso', scenario_data, config)

            if config['fed_imp']:
                print('\n\n-------- FEDERATED TRAINING  --------')

                gen_path = config[
                               'gen_data_path'] + dataset + os.sep + scenario + os.sep + col + os.sep + 'fed_imp' + os.sep
                fed_imp_results = Parallel(n_jobs=config['n_threads'], verbose=10)(
                    delayed(train_fed_imp)(scenario_data, seed, output_path, gen_path, config) for seed in
                    range(config['n_seeds']))

                # Save results
                results = save_results(fed_imp_results, results, 'fed_imp', scenario_data, config)

            if config['fed_imp_syn_naive']:
                print('\n\n-------- FEDERATED TRAINING WITH SYNTHETIC DATA (NAIVE)  --------')

                gen_path = config[
                               'gen_data_path'] + dataset + os.sep + scenario + os.sep + col + os.sep + 'fed_imp_syn_naive' + os.sep
                fed_imp_syn_naive_results = Parallel(n_jobs=config['n_threads'], verbose=10)(
                    delayed(train_fed_imp_syn_naive)(scenario_data, seed, output_path, gen_path, config) for seed in
                    range(config['n_seeds']))

                # Save results
                results = save_results(fed_imp_syn_naive_results, results, 'fed_imp_syn_naive', scenario_data, config)

            if config['fed_imp_syn_bias']:
                print('\n\n-------- FEDERATED TRAINING WITH SYNTHETIC DATA (BIAS)  --------')

                gen_path = config[
                               'gen_data_path'] + dataset + os.sep + scenario + os.sep + col + os.sep + 'fed_imp_syn_bias' + os.sep
                fed_imp_syn_bias_results = Parallel(n_jobs=config['n_threads'], verbose=10)(
                    delayed(train_fed_imp_syn_bias)(scenario_data, seed, output_path, gen_path, config) for seed in
                    range(config['n_seeds']))

                # Save results
                results = save_results(fed_imp_syn_bias_results, results, 'fed_imp_syn_bias', scenario_data, config)

            # Save results
            print('Saving results...')
            save(results, output_path + 'results_information.pickle')

    # Show results
    for dataset in config['datasets']:
        print('\n' + '-' * 50)
        print(f'\n{dataset} dataset')

        # Configure parameters
        scenario = 'scenario_7'
        col = config['pred_cols'][dataset]
        seed_eval = config['seed_eval']
        output_path = config['output_path'] + dataset + os.sep + 'scenario_7' + os.sep + col + os.sep

        # Show results
        results = check_file(output_path + 'results_information.pickle', 'Results file does not exist.')
        context = ['iso', 'fed_imp', 'fed_imp_syn_naive', 'fed_imp_syn_bias']
        best_results = show_best_results_sc_7(results, output_path, scenario, config, dataset, context,
                                              seeds_eval=seed_eval, no_col_drop=True)

        # Show results in tables to compare
        print('\n\n-------- RESULTS COMPARISON --------')
        ci_metrics = ['ci']
        ibs_metrics = ['ibs']
        pv_th = 0.05
        n_nodes = 2
        tab_ci = []
        tab_ibs = []
        uncorrected_ci_p_values = []
        uncorrected_ibs_p_values = []
        for node in range(n_nodes):

            if node == 1:
                ci_metrics.append('no_col_drop_ci')
                ibs_metrics.append('no_col_drop_ibs')
            for ci_m, ibs_m in zip(ci_metrics, ibs_metrics):
                if node == 0:
                    t_ci = ['Node ' + str(node), ci_m]
                    t_ibs = ['Node ' + str(node), ibs_m]
                else:
                    t_ci = ['Node ' + str(node) if ci_m == 'ci' else '', ci_m]
                    t_ibs = ['Node ' + str(node) if ibs_m == 'ibs' else '', ibs_m]
                for c in context:
                    if c == 'iso' and ci_m == 'no_col_drop_ci':
                        t_ci.extend(['-'])
                        t_ibs.extend(['-'])
                        continue
                    ci = str(format(best_results[c][ci_m + '_' + ibs_m]['node_' + str(node)][0], '.3f'))
                    ci_min = str(format(best_results[c][ci_m + '_' + ibs_m]['node_' + str(node)][1], '.3f'))
                    ci_max = str(format(best_results[c][ci_m + '_' + ibs_m]['node_' + str(node)][2], '.3f'))
                    t_ci.extend(['(' + ci_min + ' - ' + ci + ' - ' + ci_max + ')'])

                    ibs = str(format(best_results[c][ci_m + '_' + ibs_m]['node_' + str(node)][4], '.3f'))
                    ibs_min = str(format(best_results[c][ci_m + '_' + ibs_m]['node_' + str(node)][5], '.3f'))
                    ibs_max = str(format(best_results[c][ci_m + '_' + ibs_m]['node_' + str(node)][6], '.3f'))
                    t_ibs.extend(['(' + ibs_min + ' - ' + ibs + ' - ' + ibs_max + ')'])

                # Obtain p-values for federated models (iso is the reference)
                ci_iso = best_results['iso']['ci_ibs']['node_' + str(node)][0]
                seeds_cis_iso = [val[1] for val in best_results['iso']['ci_ibs']['node_' + str(node)][3]]
                n_obs_iso = len(seeds_cis_iso)  # seed eval
                std_iso = np.std(seeds_cis_iso)

                ci_imp = best_results['fed_imp'][ci_m + '_' + ibs_m]['node_' + str(node)][0]
                seeds_cis_imp = [val[1] for val in best_results['fed_imp'][ci_m + '_' + ibs_m]['node_' + str(node)][3]]
                n_obs_imp = len(seeds_cis_imp)
                std_imp = np.std(seeds_cis_imp)
                imp_test_ci = ttest_ind_from_stats(ci_iso, std_iso, n_obs_iso, ci_imp, std_imp, n_obs_imp,
                                                   equal_var=False, alternative='less')

                ci_synt_n = best_results['fed_imp_syn_naive'][ci_m + '_' + ibs_m]['node_' + str(node)][0]
                seeds_cis_synt_n = [val[1] for val in
                                    best_results['fed_imp_syn_naive'][ci_m + '_' + ibs_m]['node_' + str(node)][3]]
                n_obs_synt_n = len(seeds_cis_synt_n)
                std_synt_n = np.std(seeds_cis_synt_n)
                synt_test_ci_n = ttest_ind_from_stats(ci_iso, std_iso, n_obs_iso, ci_synt_n, std_synt_n,
                                                      n_obs_synt_n, equal_var=False, alternative='less')

                ci_synt_b = best_results['fed_imp_syn_bias'][ci_m + '_' + ibs_m]['node_' + str(node)][0]
                seeds_cis_synt_b = [val[1] for val in
                                    best_results['fed_imp_syn_bias'][ci_m + '_' + ibs_m]['node_' + str(node)][3]]
                n_obs_synt_b = len(seeds_cis_synt_b)
                std_synt_b = np.std(seeds_cis_synt_b)
                synt_test_ci_b = ttest_ind_from_stats(ci_iso, std_iso, n_obs_iso, ci_synt_b, std_synt_b,
                                                      n_obs_synt_b, equal_var=False, alternative='less')

                ibs_iso = best_results['iso']['ci_ibs']['node_' + str(node)][4]
                seeds_ibs_iso = [val[1] for val in best_results['iso']['ci_ibs']['node_' + str(node)][7]]
                n_obs_ibs_iso = len(seeds_ibs_iso)  # seed eval
                std_ibs_iso = np.std(seeds_ibs_iso)

                ibs_imp = best_results['fed_imp'][ci_m + '_' + ibs_m]['node_' + str(node)][4]
                seeds_ibs_imp = [val[1] for val in best_results['fed_imp'][ci_m + '_' + ibs_m]['node_' + str(node)][7]]
                n_obs_ibs_imp = len(seeds_ibs_imp)
                std_ibs_imp = np.std(seeds_ibs_imp)
                imp_test_ibs = ttest_ind_from_stats(ibs_iso, std_ibs_iso, n_obs_ibs_iso, ibs_imp, std_ibs_imp,
                                                    n_obs_ibs_imp,
                                                    equal_var=False, alternative='greater')

                ibs_synt_n = best_results['fed_imp_syn_naive'][ci_m + '_' + ibs_m]['node_' + str(node)][4]
                seeds_ibs_synt_n = [val[1] for val in
                                    best_results['fed_imp_syn_naive'][ci_m + '_' + ibs_m]['node_' + str(node)][7]]
                n_obs_ibs_synt_n = len(seeds_ibs_synt_n)
                std_ibs_synt_n = np.std(seeds_ibs_synt_n)
                synt_test_ibs_n = ttest_ind_from_stats(ibs_iso, std_ibs_iso, n_obs_ibs_iso, ibs_synt_n, std_ibs_synt_n,
                                                       n_obs_ibs_synt_n, equal_var=False, alternative='greater')

                ibs_synt_b = best_results['fed_imp_syn_bias'][ci_m + '_' + ibs_m]['node_' + str(node)][4]
                seeds_ibs_synt_b = [val[1] for val in
                                    best_results['fed_imp_syn_bias'][ci_m + '_' + ibs_m]['node_' + str(node)][7]]
                n_obs_ibs_synt_b = len(seeds_ibs_synt_b)
                std_ibs_synt_b = np.std(seeds_ibs_synt_b)
                synt_test_ibs_b = ttest_ind_from_stats(ibs_iso, std_ibs_iso, n_obs_ibs_iso, ibs_synt_b, std_ibs_synt_b,
                                                       n_obs_ibs_synt_b, equal_var=False, alternative='greater')
                # P-values
                t_ci.extend([str(format(imp_test_ci.pvalue, '.3f')) + ' / ' + str(
                    format(synt_test_ci_n.pvalue, '.3f')) + ' / ' + str(format(synt_test_ci_b.pvalue, '.3f'))])
                t_ci.extend([('*' if imp_test_ci.pvalue < pv_th else '-') + ' / ' + (
                    '*' if synt_test_ci_n.pvalue < pv_th else '-') + ' / ' + (
                                 '*' if synt_test_ci_b.pvalue < pv_th else '-')])
                # if node == 1 and ci_m == 'ci':
                if ci_m == 'ci':
                    pass
                else:
                    uncorrected_ci_p_values.extend([imp_test_ci.pvalue, synt_test_ci_n.pvalue, synt_test_ci_b.pvalue])

                t_ibs.extend([str(format(imp_test_ibs.pvalue, '.3f')) + ' / ' + str(
                    format(synt_test_ibs_n.pvalue, '.3f')) + ' / ' + str(format(synt_test_ibs_b.pvalue, '.3f'))])
                t_ibs.extend([('*' if imp_test_ibs.pvalue < pv_th else '-') + ' / ' + (
                    '*' if synt_test_ibs_n.pvalue < pv_th else '-') + ' / ' + (
                                  '*' if synt_test_ibs_b.pvalue < pv_th else '-')])
                # if node == 1 and ci_m == 'ci':
                if ci_m == 'ci':
                    pass
                else:
                    uncorrected_ibs_p_values.extend(
                        [imp_test_ibs.pvalue, synt_test_ibs_n.pvalue, synt_test_ibs_b.pvalue])

                tab_ci.append(t_ci)
                tab_ibs.append(t_ibs)

        names = ['Nodes', 'Metric', 'Isolated', 'Federated Imputation', 'Federated Imputation Synthetic Data',
                 'Federated Imputation Synthetic Data Bias', 'Significant Advantage', 'p_value < ' + str(pv_th) + ' ?']
        print('\n')
        print(tabulate(tab_ci, headers=names, tablefmt='orgtbl'))
        print('\n')
        print(tabulate(tab_ibs, headers=names, tablefmt='orgtbl'))

        # P VALUE ADJUSTMENT BY DATASET
        data_adjusted_p_values = []
        adjusted_ci_p_vals = list(multipletests(uncorrected_ci_p_values, method='holm')[1])
        adjusted_ibs_p_vals = list(multipletests(uncorrected_ibs_p_values, method='holm')[1])

        ci_metrics = ['ci']
        ibs_metrics = ['ibs']
        for node in range(n_nodes):
            if node == 0:
                continue
            elif node == 1:
                ci_metrics = ['no_col_drop_ci']
                ibs_metrics = ['no_col_drop_ibs']
            for ci_m, ibs_m in zip(ci_metrics, ibs_metrics):
                row_ci_adj_p_vals = adjusted_ci_p_vals[:3]
                del adjusted_ci_p_vals[0]
                del adjusted_ci_p_vals[0]
                del adjusted_ci_p_vals[0]
                t_adj = ['Node ' + str(node), ci_m + '_' + ibs_m,
                         str(format(row_ci_adj_p_vals[0], '.3f')) + ' / ' + str(
                             format(row_ci_adj_p_vals[1], '.3f')) + ' / ' + str(format(row_ci_adj_p_vals[2], '.3f'))]
                t_adj.extend([('*' if row_ci_adj_p_vals[0] < pv_th else '-') + ' / ' + (
                    '*' if row_ci_adj_p_vals[1] < pv_th else '-') + ' / ' + (
                                  '*' if row_ci_adj_p_vals[2] < pv_th else '-')])
                row_ibs_adj_p_vals = adjusted_ibs_p_vals[:3]
                del adjusted_ibs_p_vals[0]
                del adjusted_ibs_p_vals[0]
                del adjusted_ibs_p_vals[0]
                t_adj.extend([str(format(row_ibs_adj_p_vals[0], '.3f')) + ' / ' + str(
                    format(row_ibs_adj_p_vals[1], '.3f')) + ' / ' + str(format(row_ibs_adj_p_vals[2], '.3f'))])
                t_adj.extend([('*' if row_ibs_adj_p_vals[0] < pv_th else '-') + ' / ' + (
                    '*' if row_ibs_adj_p_vals[1] < pv_th else '-') + ' / ' + (
                                  '*' if row_ibs_adj_p_vals[2] < pv_th else '-')])

                data_adjusted_p_values.append(t_adj)

        print('\n\nADJUSTED BY DATASET')
        names = ['Nodes', 'Metrics', 'Adjusted Significant Advantage (CI)',
                 'adjusted p_value < ' + str(pv_th) + ' ?  (CI)', 'Adjusted Significant Advantage (IBS)',
                 'adjusted p_value < ' + str(pv_th) + ' ?  (IBS)']
        print(tabulate(data_adjusted_p_values, headers=names, tablefmt='orgtbl'))
        print('\n')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
