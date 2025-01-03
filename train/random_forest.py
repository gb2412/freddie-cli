
from pathlib import Path
import pickle
import polars as pl
from typing import List

from sklearn.ensemble import RandomForestClassifier

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

    log_info("END loading training data")

    return X_train_df, y_train_df


def train_random_forest(data_path: Path,
                        models_path: Path,
                        n_estimators:int,
                        max_depth:int,
                        class_weight:str,
                        monotonic_cst:List[List[str]]):
    '''
    Train Random Forest Classifier.

    Args:
        - data_path (Path): Path to samples directory
        - models_path (Path): Path to save models
        - n_estimators (int): Number of trees in the forest
        - max_depth (int): Maximum depth of each tree
        - class_weight (str): Weights associated with classes
        - monotonic_cst (List[List[str]]): List of lists of column names with monotonic constraints

    '''
    
    # Import training data
    X_train, y_train = get_train_data(data_path)

    # Define list of monotonicity constraints
    monotonicity_constraints = []
    # Loop through each column and check if it has a monotonic constraint
    for colname in X_train.columns:
        if colname in monotonic_cst['increasing']:
            monotonicity_constraints.append(-1)
        elif colname in monotonic_cst['decreasing']:
            monotonicity_constraints.append(1)
        else:
            monotonicity_constraints.append(0)

    # Create a random forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, 
                                           max_depth=max_depth,
                                           class_weight=class_weight,
                                           monotonic_cst = monotonicity_constraints)

    # Fit the best estimator on the training data
    log_info("START fitting Random Forest Classifier")
    rf_classifier.fit(X_train, y_train)

    #Save model with pickle
    with open(models_path / f'random_forest{"_balanced_" if class_weight=="balanced" else ""}{"_mon_cst" if monotonic_cst else ""}.pkl', 'wb') as file:
        pickle.dump(rf_classifier, file)

    log_info(f"END fitting Random Forest Classifier")
