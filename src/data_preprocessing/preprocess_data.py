# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 29/07/2024

# Packages to import
import os
import sys

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from src.utils import run_config, create_output_dir
from datasets import preprocess_pycox_metabric, preprocess_pycox_gbsg


def preprocess_data(dataset_name, config):
    if dataset_name == 'metabric':
        raw_data, data = preprocess_pycox_metabric()
    elif dataset_name == 'gbsg':
        raw_data, data = preprocess_pycox_gbsg()
    else:
        raise RuntimeError('Dataset not recognized')
    return raw_data, data


def main():
    print('\n\n-------- DATA PREPROCESSING  --------')

    # Environment configuration
    task = 'data_preprocessing'
    config = run_config(task)
    create_output_dir(config, task)

    # Preprocess data
    for dataset_name in config['datasets']:
        print(f'Preprocessing {dataset_name} dataset...')

        # Load dataset
        raw_data, data = preprocess_data(dataset_name, config)

        # Save data
        raw_data.to_csv(config['output_path'] + dataset_name + os.sep + 'raw_data.csv', index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
