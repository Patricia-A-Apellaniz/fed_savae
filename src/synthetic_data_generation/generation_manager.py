# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 09/08/2024


# Packages to import
import os
import torch

import pandas as pd

from joblib import Parallel, delayed
from src.synthetic_data_generation.generator import Generator
from src.utils import save, transform_data, postprocess_gen_data
from src.synthetic_data_generation.validation import discriminative_validation, utility_validation


def train_vae_bgm(data, raw_df, col_dist, seed, output_path, config, n_gen=None, tech_step=None, weights_path=None):
    # Model parameters
    latent_dim = config['gen_params']['latent_dim']
    hidden_size = config['gen_params']['hidden_size']
    model_params = {'feat_distributions': col_dist,
                    'latent_dim': latent_dim,
                    'hidden_size': hidden_size,
                    'input_dim': data[0].shape[1],
                    'early_stop': config['gen_early_stop']}
    model_path = 'gen_seed_' + str(seed)
    log_name = output_path + model_path + os.sep + tech_step + '_model' if tech_step is not None else output_path + model_path + os.sep + 'model'
    model = Generator(model_params)

    # Train the base_model
    if config['gen_train_vae']:
        train_params = {'n_epochs': config['gen_n_epochs'],
                        'batch_size': config['gen_batch_size'],
                        'device': torch.device('cpu'),
                        'lr': config['lr'],
                        'path_name': log_name}

        # If use_pre_train_weights is True, load pre-trained weights
        if weights_path is not None:
            model.load_state_dict(torch.load(weights_path))

        training_results = model.fit(data, train_params)

        # Save base_model information
        model.save(log_name)
        model_params.update(train_params)
        model_params.update(training_results)
        save(model_params, log_name + '.pickle')

    else:
        # Load already trained VAE model
        model.load_state_dict(torch.load(log_name))

    # Obtain and save synthetic samples using training data
    if n_gen is None or n_gen > 0:  # Generate only if n_gen is not negative
        n_gen = data[0].shape[0] if n_gen is None else n_gen
        model.train_latent_generator(data[0])
        model.generate(n_gen=n_gen)

        # # Validate synthetic samples
        # Denormalize generated samples to check data ranges
        model.gen_info = postprocess_gen_data(model.gen_info, raw_df)

        # Save data generated as csv
        gen_data_path = tech_step + '_processed_generated_data.csv' if tech_step is not None else 'processed_generated_data.csv'
        model.gen_info['cov_samples'].to_csv(output_path + model_path + os.sep + gen_data_path, index=False)

    return


def model_average_generation(data, raw_df, col_dist, output_path, config):
    # 1. Generate synthetic samples for each seed using train_df.shape[0] samples from real data
    model_avg_step_1 = 'first_step'
    Parallel(n_jobs=1, verbose=10)(delayed(train_vae_bgm)(data, raw_df, col_dist, seed, output_path, config, tech_step=model_avg_step_1, n_gen=-1) for seed in range(config['n_seeds'])) # TODO: Juan changed n_gen to -1, not to generate in this step (not needed)

    # 2. Average weights from all seeds and train the final model
    # Load model weights from each seed
    results = {}
    for seed in range(config['n_seeds']):
        log_name = output_path + 'gen_seed_' + str(seed) + os.sep + model_avg_step_1 + '_model'
        results[seed] = torch.load(log_name)
    # Average weights
    avg_weights = {}
    for key in results[0].keys():
        avg_weights[key] = sum([results[seed][key] for seed in range(config['n_seeds'])]) / config['n_seeds']
    # Save average weights
    weights_path = output_path + 'seed_avg_weights_model'
    torch.save(avg_weights, weights_path)

    # 3. Train final model with average weights
    train_vae_bgm(data, raw_df, col_dist, 0, output_path, config, n_gen=config['n_gen'], weights_path=weights_path)


def generate(data, raw_df, col_dist, output_path, technique, config):
    if technique == 'model_avg':  # Train model with model averaging technique
        print('Training model with model average technique...')
        model_average_generation(data, raw_df, col_dist, output_path, config)

    else:
        raise NotImplementedError('Technique not recognized')
    return


def evaluate(raw_df, col_dist, output_path):
    # Load data
    gen_data = pd.read_csv(output_path + os.sep + 'gen_seed_0' + os.sep + 'processed_generated_data.csv')

    # Denormalize
    raw_real_df = raw_df.copy()
    raw_gen_df = transform_data(gen_data, raw_df, col_dist, denorm=True)

    # Evaluate generated data
    results = {'task': utility_validation(raw_real_df, raw_gen_df),
               'disc': discriminative_validation(raw_real_df, raw_gen_df, ['RF'])}

    return results
