from pathlib import Path
import pickle
import polars as pl
import numpy as np
from scipy.sparse import csc_matrix

import fastsparsegams

from utils.logging import log_info


def get_train_data(data_path: Path):
    '''
    Import training set.

    Args:
        - data_path (Path): Path to samples directory
    '''

    log_info("START loading training data")
    
    # Check if training data exists
    if not (data_path / 'train_binary.parquet').exists():
        raise FileNotFoundError(f"""File {data_path / 'train_binary.parquet'} not found.
                                Please use freddie sample --binary to sample train and test sets.""")
    # Import training data
    data_train = pl.read_parquet(data_path / 'train_binary.parquet')

    # Splitting into X and y
    X_train_df, y_train_df = data_train.drop(['Default','Loan_Num','Month']), data_train.select('Default')
    del data_train

    # Converting to numpy
    X_train = X_train_df.to_numpy().astype(np.float64)
    del X_train_df

    # Converting to numpy
    y_train = y_train_df.to_numpy().flatten().astype(np.int32)
    # Freeing memory
    del y_train_df

    # Convert into csc_matrix
    X_train = csc_matrix(X_train)

    log_info("END loading training data")

    return X_train, y_train


def train_sparse_gam(data_path: Path,
                        models_path: Path,
                        loss,
                        penalty:str,
                        max_support_size:int,
                        num_lambda:int,
                        algorithm:str,
                        scale_down_factor:float
                        ):
    '''
    Train sparse GAM model.

    Args:
        - data_path (Path): Path to samples directory
        - models_path (Path): Path to save models
        - loss (str): Loss function to use
        - penalty (str): Penalty type to use
        - max_support_size (int): Maximum support size
        - num_lambda (int): Number of lambdas
        - algorithm (str): Fitting algorithm to use
        - scale_down_factor (float): Scale down factor
    '''

    # Import training data
    X_train, y_train = get_train_data(data_path)

    log_info(f"START fitting CV {loss} loss")

    # Fit model
    fit_model_cv = fastsparsegams.fit(
        X_train, 
        y_train, 
        loss,
        penalty,
        max_support_size,
        num_lambda,
        algorithm,
        scale_down_factor
    )

    #Save model with pickle
    with open(models_path / f'sparse_GAM_{loss}.pkl', 'wb') as file:
        pickle.dump(fit_model_cv, file)

    log_info(f"END fitting CV {loss} loss")
