# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 29/07/2024

# Packages to import
import numpy as np
import pandas as pd

from pycox import datasets


def preprocess_pycox_metabric():
    # Load data
    raw_df = datasets.metabric.read_df()
    raw_df = raw_df.sample(frac=1).reset_index(drop=True)  # Reorder the data randomly in case patients are ordered

    # Transform covariates and create df
    label = raw_df[['event']]
    time = raw_df[['duration']]
    raw_df = raw_df.drop(labels=['event', 'duration'], axis=1)
    survival_df = time.join(label)
    raw_df = raw_df.join(survival_df)
    raw_df = raw_df.rename(columns={raw_df.columns[-1]: 'event', raw_df.columns[-2]: 'time'})

    # Copy dataframe to normalize it
    df = raw_df.copy()

    # Normalize df
    # 1. Select columns distributions
    col_dist = []
    for i in range(df.shape[1]):
        values = df.iloc[:, i].unique()
        no_nan_values = values[~pd.isnull(values)]
        if no_nan_values.size <= 2:
            col_dist.append(('bernoulli', 1))
        # elif i == 0 or i == 1 or i == df.shape[1]-2:
        #     col_dist.append(('log-normal', 2))
        else:
            col_dist.append(('gaussian', 2))

    # Change time distribution to weibull
    col_dist[-2] = ('weibull', 2)

    # 2. Normalize columns
    for i in range(df.shape[1]):
        if col_dist[i][0] == 'gaussian':
            loc = df.iloc[:, i].mean()
            scale = df.iloc[:, i].std()
            df.iloc[:, i] = (df.iloc[:, i] - loc) / scale
        elif col_dist[i][0] == 'log-normal':
            loc = df.iloc[:, i].mean()
            scale = df.iloc[:, i].std()
            df.iloc[:, i] = (np.log(df.iloc[:, i]) - loc) / scale
        elif col_dist[i][0] == 'bernoulli':
            pass  # Do nothing
        elif col_dist[i][0] == 'weibull':
            values = df.iloc[:, i].unique()
            no_nan_values = values[~pd.isnull(values)]
            loc = -1 if 0 in no_nan_values else 0
            df.iloc[:, i] = df.iloc[:, i] - loc
        else:
            raise NotImplementedError('Distribution ', col_dist[i][0], ' not normalized!')

    return raw_df, df


def preprocess_pycox_gbsg():
    # Load data
    raw_df = datasets.gbsg.read_df()
    raw_df = raw_df.sample(frac=1).reset_index(drop=True)  # Reorder the data randomly in case patients are ordered

    # Transform covariates and create df
    label = raw_df[['event']]
    time = raw_df[['duration']]
    raw_df = raw_df.drop(labels=['event', 'duration'], axis=1)
    survival_df = time.join(label)
    raw_df = raw_df.join(survival_df)
    raw_df = raw_df.rename(columns={raw_df.columns[-1]: 'event', raw_df.columns[-2]: 'time'})

    # Copy dataframe to normalize it
    df = raw_df.copy()

    # Normalize df
    # 1. Select columns distributions
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

    # 2. Normalize columns
    for i in range(df.shape[1]):
        if col_dist[i][0] == 'gaussian':
            loc = df.iloc[:, i].mean()
            scale = df.iloc[:, i].std()
            df.iloc[:, i] = (df.iloc[:, i] - loc) / scale
        elif col_dist[i][0] == 'bernoulli':
            pass  # Do nothing
        elif col_dist[i][0] == 'weibull':
            values = df.iloc[:, i].unique()
            no_nan_values = values[~pd.isnull(values)]
            loc = -1 if 0 in no_nan_values else 0
            df.iloc[:, i] = df.iloc[:, i] - loc
        else:
            raise NotImplementedError('Distribution ', col_dist[i][0], ' not normalized!')

    return raw_df, df
