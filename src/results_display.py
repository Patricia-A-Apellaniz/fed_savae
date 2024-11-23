# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 31/07/2024

# Packages to import
import numpy as np
import matplotlib.pyplot as plt


def show_best_results(results, output_path, scenario, config, dataset_name, modes, seeds_eval=3):
    params = str(config['params']['latent_dim']) + '_' + str(config['params']['hidden_size'])
    n_seeds = config['n_seeds']

    print('\n' + scenario + ' results:')
    node_best_results = {}
    colors = ['r', 'g', 'b']
    for mode in modes:
        print('\n\nMode: ' + mode)
        mode_results = results[mode]
        node_best_results[mode] = {}
        ci = 'ci'
        ibs = 'ibs'
        fig_ci = plt.figure(1)
        fig_ci.set_size_inches(12, 6)
        ax_ci = fig_ci.add_subplot(111)
        fig_ibs = plt.figure(2)
        fig_ibs.set_size_inches(12, 6)
        ax_ibs = fig_ibs.add_subplot(111)
        for item in mode_results.items():
            node = item[1][params]
            ci_per_seed = [node[ci][seed][-1] for seed in range(n_seeds)]
            ibs_per_seed = [node[ibs][seed][-1] for seed in range(n_seeds)]

            # Select based on both metrics
            differences = []
            best_idx = []
            for i in range(len(ci_per_seed)):
                diff = ci_per_seed[i][1] - ibs_per_seed[i][1]
                if seeds_eval > len(differences):
                    best_idx.append(i)
                    differences.append(diff)
                else:
                    min_dif_idx = np.argmin(differences)
                    if diff > differences[min_dif_idx]:
                        differences[min_dif_idx] = diff
                        best_idx[min_dif_idx] = i
            node_best_results[mode][item[0]] = (np.mean(np.array([ci[1] for ci in ci_per_seed])[best_idx]),
                                                np.min(np.array([ci[0] for ci in ci_per_seed])[best_idx]),
                                                np.max(np.array([ci[2] for ci in ci_per_seed])[best_idx]),
                                                [ci_per_seed[idx] for idx in best_idx],
                                                np.mean(np.array([ibs[1] for ibs in ibs_per_seed])[best_idx]),
                                                np.min(np.array([ibs[0] for ibs in ibs_per_seed])[best_idx]),
                                                np.max(np.array([ibs[2] for ibs in ibs_per_seed])[best_idx]),
                                                [ibs_per_seed[idx] for idx in best_idx],
                                                best_idx)

            # Results presentation
            print('Node: ' + str(item[0]))
            print('Average IBS for testing sets: ' + str(format(node_best_results[mode][item[0]][4], '.3f')))
            print('Average C-Index for testing sets: ' + str(format(node_best_results[mode][item[0]][0], '.3f')))

            # Take best seeds mean across all epochs
            best_seeds = node_best_results[mode][item[0]][8]
            seeds_epochs_avg_cis = np.array([node[ci][seed][:-1, 1] for seed in best_seeds])
            seeds_epochs_min_cis = np.array([node[ci][seed][:-1, 0] for seed in best_seeds])
            seeds_epochs_max_cis = np.array([node[ci][seed][:-1, 2] for seed in best_seeds])

            # Set to np.nan values below 0
            seeds_epochs_avg_cis[seeds_epochs_avg_cis < 0] = np.nan
            seeds_epochs_min_cis[seeds_epochs_min_cis < 0] = np.nan
            seeds_epochs_max_cis[seeds_epochs_max_cis < 0] = np.nan

            # Delete parts of the signal where there are only nans (i.e., no seed has lasted that long)
            actual_idx = np.where(np.sum(np.isnan(seeds_epochs_avg_cis), axis=0) < seeds_eval)[0]
            seeds_epochs_avg_cis = seeds_epochs_avg_cis[:, actual_idx]
            seeds_epochs_min_cis = seeds_epochs_min_cis[:, actual_idx]
            seeds_epochs_max_cis = seeds_epochs_max_cis[:, actual_idx]

            # Get average across seeds
            epochs_avg_cis = np.nanmean(seeds_epochs_avg_cis, axis=0)
            epochs_min_cis = np.nanmin(seeds_epochs_min_cis, axis=0)
            epochs_max_cis = np.nanmax(seeds_epochs_max_cis, axis=0)

            # Do the same for ibs
            best_seeds = node_best_results[mode][item[0]][8]
            seeds_epochs_avg_ibs = np.array([node[ibs][seed][:-1, 1] for seed in best_seeds])
            seeds_epochs_min_ibs = np.array([node[ibs][seed][:-1, 0] for seed in best_seeds])
            seeds_epochs_max_ibs = np.array([node[ibs][seed][:-1, 2] for seed in best_seeds])

            # Set to np.nan values below 0
            seeds_epochs_avg_ibs[seeds_epochs_avg_ibs < 0] = np.nan
            seeds_epochs_min_ibs[seeds_epochs_min_ibs < 0] = np.nan
            seeds_epochs_max_ibs[seeds_epochs_max_ibs < 0] = np.nan

            # Delete parts of the signal where there are only nans (i.e., no seed has lasted that long)
            actual_idx = np.where(np.sum(np.isnan(seeds_epochs_avg_ibs), axis=0) < seeds_eval)[0]
            seeds_epochs_avg_ibs = seeds_epochs_avg_ibs[:, actual_idx]
            seeds_epochs_min_ibs = seeds_epochs_min_ibs[:, actual_idx]
            seeds_epochs_max_ibs = seeds_epochs_max_ibs[:, actual_idx]

            # Get average across seeds for each fold
            epochs_avg_ibs = np.nanmean(seeds_epochs_avg_ibs, axis=0)
            epochs_min_ibs = np.nanmin(seeds_epochs_min_ibs, axis=0)
            epochs_max_ibs = np.nanmax(seeds_epochs_max_ibs, axis=0)

            # Average cis across folds
            n_epochs = len(epochs_avg_cis)
            node_number = int(item[0].split('_')[1])
            ax_ci.plot(np.linspace(0, n_epochs - 1, n_epochs), seeds_epochs_avg_cis[0], label=str(item[0]),
                       color=colors[node_number])
            ax_ci.plot(np.linspace(0, n_epochs - 1, n_epochs), seeds_epochs_avg_cis[1:].T,
                       color=colors[node_number])
            ax_ci.fill_between(np.linspace(0, n_epochs - 1, n_epochs), epochs_min_cis, epochs_max_cis, alpha=0.2,
                               color=colors[node_number])

            # Average ibs across folds
            ax_ibs.plot(np.linspace(0, n_epochs - 1, n_epochs), seeds_epochs_avg_ibs[0], label=str(item[0]),
                        color=colors[node_number])
            ax_ibs.plot(np.linspace(0, n_epochs - 1, n_epochs), seeds_epochs_avg_ibs[1:].T,
                        color=colors[node_number])
            ax_ibs.fill_between(np.linspace(0, n_epochs - 1, n_epochs), epochs_min_ibs, epochs_max_ibs, alpha=0.2,
                                color=colors[node_number])

        ax_ci.legend(bbox_to_anchor=(1.05, 0.015), loc='lower left', borderaxespad=0.)
        ax_ci.grid('on')
        ax_ci.set_xlabel('Epochs')
        ax_ci.set_ylabel('C-index')
        title = 'C-index during training process. ' + mode + '.' + ' ' + dataset_name
        dir = 'epochs_' + ci + '_' + mode + '.png'
        ax_ci.set_title(scenario + '. ' + title)
        fig_ci.tight_layout()
        fig_ci.savefig(output_path + dir, bbox_inches='tight')

        ax_ibs.legend(bbox_to_anchor=(1.05, 0.015), loc='lower left', borderaxespad=0.)
        ax_ibs.grid('on')
        ax_ibs.set_xlabel('Epochs')
        ax_ibs.set_ylabel('IBS')
        title = 'IBS during training process. ' + mode + '.' + ' ' + dataset_name
        dir = 'epochs_' + ibs + '_' + mode + '.png'
        ax_ibs.set_title(scenario + '. ' + title)
        fig_ibs.tight_layout()
        fig_ibs.savefig(output_path + dir, bbox_inches='tight')
        # plt.show()
        plt.close()

    return node_best_results

def show_best_results_sc_7(results, output_path, scenario, config, dataset_name, modes, seeds_eval=3, no_col_drop=False):
    params = str(config['params']['latent_dim']) + '_' + str(config['params']['hidden_size'])
    n_seeds = config['n_seeds']

    print('\n' + scenario + ' results:')
    node_best_results = {}
    colors = ['r', 'g', 'b']
    for mode in modes:
        print('\n\nMode: ' + mode)
        mode_results = results[mode]
        node_best_results[mode] = {}
        ci_metrics = ['ci']
        ibs_metrics = ['ibs']
        if no_col_drop and mode != 'iso':
            ci_metrics.append('no_col_drop_ci')
            ibs_metrics.append('no_col_drop_ibs')
        for ci, ibs in zip(ci_metrics, ibs_metrics):
            node_best_results[mode][ci + '_' + ibs] = {}
            fig_ci = plt.figure(1)
            fig_ci.set_size_inches(12, 6)
            ax_ci = fig_ci.add_subplot(111)
            fig_ibs = plt.figure(2)
            fig_ibs.set_size_inches(12, 6)
            ax_ibs = fig_ibs.add_subplot(111)
            for item in mode_results.items():
                node = item[1][params]
                ci_per_seed = [node[ci][seed][-1] for seed in range(n_seeds)]
                ibs_per_seed = [node[ibs][seed][-1] for seed in range(n_seeds)]

                # Select based on both metrics
                differences = []
                best_idx = []
                for i in range(len(ci_per_seed)):
                    diff = ci_per_seed[i][1] - ibs_per_seed[i][1]
                    if seeds_eval > len(differences):
                        best_idx.append(i)
                        differences.append(diff)
                    else:
                        min_dif_idx = np.argmin(differences)
                        if diff > differences[min_dif_idx]:
                            differences[min_dif_idx] = diff
                            best_idx[min_dif_idx] = i
                node_best_results[mode][ci + '_' + ibs][item[0]] = (np.mean(np.array([ci[1] for ci in ci_per_seed])[best_idx]),
                                                    np.min(np.array([ci[0] for ci in ci_per_seed])[best_idx]),
                                                    np.max(np.array([ci[2] for ci in ci_per_seed])[best_idx]),
                                                    [ci_per_seed[idx] for idx in best_idx],
                                                    np.mean(np.array([ibs[1] for ibs in ibs_per_seed])[best_idx]),
                                                    np.min(np.array([ibs[0] for ibs in ibs_per_seed])[best_idx]),
                                                    np.max(np.array([ibs[2] for ibs in ibs_per_seed])[best_idx]),
                                                    [ibs_per_seed[idx] for idx in best_idx],
                                                    best_idx)

                # Results presentation
                if ci == 'ci':
                    print('\nNode: ' + str(item[0]))
                    print('Average IBS for testing sets: ' + str(format(node_best_results[mode][ci + '_' + ibs][item[0]][4], '.3f')))
                    print('Average C-Index for testing sets: ' + str(format(node_best_results[mode][ci + '_' + ibs][item[0]][0], '.3f')))
                if item[0] == 'node_1' and ci == 'no_col_drop_ci':
                    print('Average IBS for no-col-drop-testing sets: ' + str(
                        format(node_best_results[mode][ci + '_' + ibs][item[0]][4], '.3f')))
                    print('Average C-Index for no-col-drop-testing sets: ' + str(
                        format(node_best_results[mode][ci + '_' + ibs][item[0]][0], '.3f')))


                # Take best seeds mean across all epochs
                best_seeds = node_best_results[mode][ci + '_' + ibs][item[0]][8]
                seeds_epochs_avg_cis = np.array([node[ci][seed][:-1, 1] for seed in best_seeds])
                seeds_epochs_min_cis = np.array([node[ci][seed][:-1, 0] for seed in best_seeds])
                seeds_epochs_max_cis = np.array([node[ci][seed][:-1, 2] for seed in best_seeds])

                # Set to np.nan values below 0
                seeds_epochs_avg_cis[seeds_epochs_avg_cis < 0] = np.nan
                seeds_epochs_min_cis[seeds_epochs_min_cis < 0] = np.nan
                seeds_epochs_max_cis[seeds_epochs_max_cis < 0] = np.nan

                # Delete parts of the signal where there are only nans (i.e., no seed has lasted that long)
                actual_idx = np.where(np.sum(np.isnan(seeds_epochs_avg_cis), axis=0) < seeds_eval)[0]
                seeds_epochs_avg_cis = seeds_epochs_avg_cis[:, actual_idx]
                seeds_epochs_min_cis = seeds_epochs_min_cis[:, actual_idx]
                seeds_epochs_max_cis = seeds_epochs_max_cis[:, actual_idx]

                # Get average across seeds
                epochs_avg_cis = np.nanmean(seeds_epochs_avg_cis, axis=0)
                epochs_min_cis = np.nanmin(seeds_epochs_min_cis, axis=0)
                epochs_max_cis = np.nanmax(seeds_epochs_max_cis, axis=0)

                # Do the same for ibs
                best_seeds = node_best_results[mode][ci + '_' + ibs][item[0]][8]
                seeds_epochs_avg_ibs = np.array([node[ibs][seed][:-1, 1] for seed in best_seeds])
                seeds_epochs_min_ibs = np.array([node[ibs][seed][:-1, 0] for seed in best_seeds])
                seeds_epochs_max_ibs = np.array([node[ibs][seed][:-1, 2] for seed in best_seeds])

                # Set to np.nan values below 0
                seeds_epochs_avg_ibs[seeds_epochs_avg_ibs < 0] = np.nan
                seeds_epochs_min_ibs[seeds_epochs_min_ibs < 0] = np.nan
                seeds_epochs_max_ibs[seeds_epochs_max_ibs < 0] = np.nan

                # Delete parts of the signal where there are only nans (i.e., no seed has lasted that long)
                actual_idx = np.where(np.sum(np.isnan(seeds_epochs_avg_ibs), axis=0) < seeds_eval)[0]
                seeds_epochs_avg_ibs = seeds_epochs_avg_ibs[:, actual_idx]
                seeds_epochs_min_ibs = seeds_epochs_min_ibs[:, actual_idx]
                seeds_epochs_max_ibs = seeds_epochs_max_ibs[:, actual_idx]

                # Get average across seeds for each fold
                epochs_avg_ibs = np.nanmean(seeds_epochs_avg_ibs, axis=0)
                epochs_min_ibs = np.nanmin(seeds_epochs_min_ibs, axis=0)
                epochs_max_ibs = np.nanmax(seeds_epochs_max_ibs, axis=0)

                # Average cis across folds
                n_epochs = len(epochs_avg_cis)
                node_number = int(item[0].split('_')[1])
                ax_ci.plot(np.linspace(0, n_epochs - 1, n_epochs), seeds_epochs_avg_cis[0], label=str(item[0]),
                           color=colors[node_number])
                ax_ci.plot(np.linspace(0, n_epochs - 1, n_epochs), seeds_epochs_avg_cis[1:].T,
                           color=colors[node_number])
                ax_ci.fill_between(np.linspace(0, n_epochs - 1, n_epochs), epochs_min_cis, epochs_max_cis, alpha=0.2,
                                   color=colors[node_number])

                # Average ibs across folds
                ax_ibs.plot(np.linspace(0, n_epochs - 1, n_epochs), seeds_epochs_avg_ibs[0], label=str(item[0]),
                            color=colors[node_number])
                ax_ibs.plot(np.linspace(0, n_epochs - 1, n_epochs), seeds_epochs_avg_ibs[1:].T,
                            color=colors[node_number])
                ax_ibs.fill_between(np.linspace(0, n_epochs - 1, n_epochs), epochs_min_ibs, epochs_max_ibs, alpha=0.2,
                                    color=colors[node_number])

            ax_ci.legend(bbox_to_anchor=(1.05, 0.015), loc='lower left', borderaxespad=0.)
            ax_ci.grid('on')
            ax_ci.set_xlabel('Epochs')
            ax_ci.set_ylabel('C-index')
            title = 'C-index (' + ci + ') during training process. ' + mode + '.' + ' ' + dataset_name
            dir = 'epochs_' + ci + '_' + mode + '.png'
            ax_ci.set_title(scenario + '. ' + title)
            fig_ci.tight_layout()
            fig_ci.savefig(output_path + dir, bbox_inches='tight')

            ax_ibs.legend(bbox_to_anchor=(1.05, 0.015), loc='lower left', borderaxespad=0.)
            ax_ibs.grid('on')
            ax_ibs.set_xlabel('Epochs')
            ax_ibs.set_ylabel('IBS')
            title = 'IBS (' + ibs + ') during training process. ' + mode + '.' + ' ' + dataset_name
            dir = 'epochs_' + ibs + '_' + mode + '.png'
            ax_ibs.set_title(scenario + '. ' + title)
            fig_ibs.tight_layout()
            fig_ibs.savefig(output_path + dir, bbox_inches='tight')
            # plt.show()
            plt.close()

    return node_best_results

