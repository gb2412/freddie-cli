from pathlib import Path
from typing import List, Tuple

import polars as pl
import numpy as np

from utils.logging import Colors, log_info
from process.economic_data_processing import get_merged_econ_data, add_econ_features


def sample_set(df: pl.DataFrame,
               obs_dates: list,
               size: int,
               seed: int) -> pl.DataFrame:
    
    # Filter for relevant observation dates
    df = df.filter(pl.col('Month').dt.strftime('%Y-%m').is_in(obs_dates))

    # Sample from each observation date if required
    if size:

        # Partition by observation date
        df_parts = df.partition_by('Month')

        # Sample from each partition and concatenate
        # sample entire partition if size > partition size
        df = pl.concat([part.sample(
                            min(size, part.shape[0]), 
                            seed=seed) 
                        for part in df_parts])

    return df


def sample_train_test_sets(data_dir: Path,
                           years_quarters_list: List[Tuple[int, int]],
                           train_obs_dates: list,
                           test_obs_dates: list,
                           train_size: int,
                           test_size: int,
                           seed: int) -> Tuple[pl.DataFrame, pl.DataFrame]:
    '''
    Sample the year-quarter combinations and observation dates.
    Exclude mortgages in default or with zero balance at the observation date.

    Args:
        - data_dir (Path): Path to the processed data directory.
        - years_quarters_list (list): List of tuples with year and quarter to sample.
        - obs_dates (list): List of observation dates to sample.
    
    '''

    log_info(f"{Colors.OKGREEN}Start sample import for {len(years_quarters_list)} quarters{Colors.ENDC}")

    # Definine training and test data frames
    df_train, df_test = pl.DataFrame(), pl.DataFrame()

    # Definie training and test sample size:
    # get equal number of loans for each quarter and observation date
    # if None entire sample is used
    train_size = train_size // len(years_quarters_list) // len(train_obs_dates) if train_size else None
    test_size = test_size // len(years_quarters_list) // len(test_obs_dates) if test_size else None

    # Loop over year-quarter combinations
    for year, quarter in years_quarters_list:
        try:
            df_temp = pl.read_parquet(data_dir / f'{year}Q{quarter}')
        except Exception as e:
            raise RuntimeError(f"Error loading {year}Q{quarter}: {str(e)}")

        # Select the sample
        df_temp = df_temp.filter(
            # Filter only relevant observation dates
            (pl.col('Month').dt.strftime('%Y-%m').is_in(train_obs_dates + test_obs_dates) &
            # Filter only mortgages that have NOT incurred in any Termination Event at the respective observation date
            pl.col('Zero_Balance_Code').is_null() &
            # Filter only 0 DPD, 30 DPD or 60 DPD
            pl.col('Current_Delinquency_Status').is_in(['0','1','2']))
        )
        
        # Sample training set
        df_train_temp = sample_set(df_temp, train_obs_dates, train_size, seed)
        # Concatenate to the main training set
        df_train = pl.concat([df_train, df_train_temp])

        # Sample test set
        df_test_temp = sample_set(df_temp, test_obs_dates, test_size, seed)
        # Concatenate to the main test set
        df_test = pl.concat([df_test, df_test_temp])

    log_info(f"Train sample shape: {df_train.shape}, Test sample shape: {df_test.shape}")

    return df_train, df_test


def add_economic_features(df: pl.DataFrame,
                          econ_data_path: Path) -> pl.DataFrame:
    '''
    Add economic features to the sample.

    Args:
        - df (pl.DataFrame): Loan data sample.
        - econ_data_path (Path): Path to the economic data.
    '''
    
    # Add economic data
    merged_df = get_merged_econ_data(econ_data_path, df)
    # Add new economic and combined features
    merged_df = add_econ_features(merged_df)

    log_info(f"Added economic features to the sample, new sample shape: {merged_df.shape}")

    return merged_df


def features_selection(df: pl.DataFrame,
                       col_list: List[str]) -> pl.DataFrame:
    '''
    Select relevant features from the sample.

    Args:
        - df (pl.DataFrame): Loan data sample.
        - col_list (list): List of columns to select.
    '''

    # Select sample columns
    df = df.select(col_list)

    log_info(f"Selected {len(col_list)} features from the sample, new sample shape: {df.shape}")

    return df


def train_test_split(df: pl.DataFrame,
                     train_obs_dates: List[str],
                     test_obs_dates: List[str],
                     train_size: int) -> Tuple[pl.DataFrame, pl.DataFrame]:
    '''
    Split training and test sets based on respective observation dates.
    The training set is composed by an equal number of loans for each observation date,
    summing up to the train size.

    Args:
        - df (pl.DataFrame): Loan data sample.
        - train_obs_dates (list): Observation dates of training set.
        - test_obs_dates (list): Observation dates of test set.
        - train_size (int): Number of loans to sample for training set.
    '''

    # Get train and test data
    data_train = df.filter(pl.col('Month').dt.strftime('%Y-%m').is_in(train_obs_dates))
    data_test = df.filter(pl.col('Month').dt.strftime('%Y-%m').is_in(test_obs_dates))

    # Divide train size by number of observation dates
    train_size = train_size // len(train_obs_dates)

    # Initialize empty dataframe with same schema as data_train
    data_train_sampled = pl.DataFrame(schema=data_train.schema)

    # Sample from each month and concat
    for month in train_obs_dates:
        month_sample = (
            data_train
            .filter(pl.col('Month').dt.strftime('%Y-%m') == month)
            .sample(n=train_size)
        )
        data_train_sampled = data_train_sampled.vstack(month_sample)
    
    log_info(f"Train sample shape: {data_train_sampled.shape}, Test sample shape: {data_test.shape}")

    return data_train_sampled, data_test


def missing_values_encoding(df: pl.DataFrame) -> pl.DataFrame:
    '''
    Encode missing values as new binary features.

    Args:
        - df (pl.DataFrame): Loan data sample.
    '''

    # Get list of columns with any missing values
    cols_with_nulls = (
        df.null_count()
        .unpivot(variable_name='col_name',value_name='null_count')
        .filter(pl.col('null_count') > 0)
        .select('col_name')
        .to_series()
        .to_list() 
    )

    # Create missing indicators only for columns with nulls
    missing_indicators = (
        df
        .select(cols_with_nulls)
        .with_columns([
            pl.col(col).is_null().cast(pl.Int8).alias(f'{col} is Missing')
            for col in cols_with_nulls
        ])
        .drop(cols_with_nulls)
    )

    # Rename special columns
    if 'Credit_Score' in cols_with_nulls:
        missing_indicators = missing_indicators.rename(
            {'Credit_Score is Missing': 'Credit_Score <= 300'}
        )
    if 'MI_Percentage' in cols_with_nulls:
        missing_indicators = missing_indicators.rename(
            {'MI_Percentage is Missing': 'MI_Percentage not in 1-55'}
        )

    # Combine original df with missing indicators
    df = pl.concat([df, missing_indicators], how='horizontal')

    log_info(f"Added missing indicators to the sample, new sample shape: {df.shape}")

    return df


def binary_enc(col_name: str, 
               value: str, 
               alias: str) -> pl.Expr:
    '''
    Get binary encoding expression for a categorical column.

    Args:
        - col_name (str): Column name.
        - value (str): Value to encode.
        - alias (str): New column name.
    '''

    return pl.when(pl.col(col_name) == value).then(1).otherwise(0).alias(alias)


def y_n_encoding(column):
    '''
    Get binary encoding expression for a Y/N column.

    Args:
        - column (str): Column name.
    '''
    return binary_enc(column, 'Y', f'{column} = Yes')


def cat_var_encoding(df: pl.DataFrame,
                     categorical_encodings: dict) -> pl.DataFrame:
    '''
    Apply binary encoding to categorical variables.

    Args:
        - df (pl.DataFrame): Loan data sample.
    '''
    
    # Get columns mappings
    yn_columns = categorical_encodings['yn_columns']
    mappings = categorical_encodings['categorical_mappings']

    # Combine all encodings
    encodings = (
        # Y/N encodings
        [y_n_encoding(col) for col in categorical_encodings['yn_columns']] +
        # Custom mappings
        [binary_enc(col, val, f'{col} = {name}') for col, mapping in mappings.items() for val, name in mapping]
    )

    log_info(f"Turned categorical features into binary, new sample shape: {df.shape}")

    return df.with_columns(encodings).drop(yn_columns+list(mappings.keys()))


def create_thresholds(series: pl.Series,
                      num_thresholds: int,
                      is_credit_score: bool) -> np.ndarray:
    '''
    Create percentile thresholds for a continuous variable.

    Args:
        - series (pl.Series): Continuous variable.
        - num_thresholds (int): Number of thresholds to create.
        - is_credit_score (bool): Whether the variable is credit score.
    '''
        
    # Remove nulls and get percentiles
    thresholds = np.percentile(series.drop_nulls().to_numpy(), q=np.arange(5, 100, round(100/num_thresholds)))

    # Fro credit score, round to the nearest lower 50 for interpretability
    if is_credit_score:
        return np.unique(thresholds // 50 * 50)
    
    # Round to 2 decimal places
    return np.unique(np.round(thresholds, 2))


def cont_var_encoding(df: pl.DataFrame,
                      num_thresholds :int) -> pl.DataFrame:
    '''
    Apply binary econding to continous columns based on percentile thresholds.

    Args:
        - df (pl.DataFrame): Loan data sample.
    '''
    
    # Store original columns
    original_cols = []
    # Generate threshold expressions for numeric columns
    threshold_exprs = []
    for col in df.columns:
        # Exclude categorical, index and target columns
        if (not df[col].dtype in [pl.Float64, pl.Int64]) | (col in ['Loan_Num','Month','Default']):
            continue

        original_cols.append(col)
        # Call create thresholds function
        thresholds = create_thresholds(
            df[col], 
            num_thresholds,
            is_credit_score=(col == 'Credit_Score')
        )
        # Create binary columns: 1 if value <= threshold, 0 otherwise
        threshold_exprs.extend([
            pl.col(col).le(threshold).cast(pl.Int8).alias(f"{col} <= {threshold}")
            for threshold in thresholds
        ])

    log_info(f"Turned continous features into binary, new sample shape: {df.shape}")

    return df.with_columns(threshold_exprs).drop(original_cols)


def save_sample_to_disk(data_train: pl.DataFrame,
                        data_test: pl.DataFrame,
                        output_path: Path,
                        binary_output: bool) -> None:
    '''
    Save samples to disk.

    Args:
        - data_train (pl.DataFrame): Training sample.
        - data_test (pl.DataFrame): Test sample.
        - output_path (Path): Path to save the samples.
        - binary_output (bool): Whether to save binary or regular samples.
    '''

    # Create output directory if it does not exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Saving datasets to parquet
    if binary_output:
        data_train.write_parquet(output_path / 'train_binary.parquet')
        data_test.write_parquet(output_path / 'test_binary.parquet')
    else:
        data_train.write_parquet(output_path / 'train.parquet')
        data_test.write_parquet(output_path / 'test.parquet')

    log_info(f"{Colors.OKGREEN}Saved train and test samples{Colors.ENDC} to {output_path}")


def process_sample(df: pl.DataFrame,
                   dev_columns: list,
                   categorical_encodings: dict,
                   num_thresholds: int,
                   binary_output: bool,
                   use_economic_data: bool,
                   econ_data_path: Path,
                   is_train_set: bool) -> pl.DataFrame:


    log_info(f"{Colors.OKGREEN}Start {'training' if is_train_set else 'test'} sample processing{Colors.ENDC}")

    # Add economic features
    if use_economic_data:
        df = add_economic_features(df, econ_data_path)

    # Select features
    df = features_selection(df, dev_columns)

    if binary_output:
        # Encode missing values
        df = missing_values_encoding(df)

        # Encode categorical variables
        df = cat_var_encoding(df, categorical_encodings)

    if binary_output:
        # Encode continuous variables
        df = cont_var_encoding(df, num_thresholds)

    return df
    

def process_and_save_train_test(data_path: Path,
                                years_quarters_list: list,
                                train_obs_dates: list,
                                test_obs_dates:list,
                                dev_columns: list,
                                train_size: int,
                                test_size: int,
                                categorical_encodings: dict,
                                num_thresholds: int,
                                output_path: Path,
                                binary_output: bool=False,
                                use_economic_data: bool=False,
                                econ_data_path: Path=None,
                                seed :int=None) -> Tuple[pl.DataFrame, pl.DataFrame]:
    '''
    Main sampling function. Load, process and save the train and test samples.
    Differentiate bewteen binary and regular sample creation.

    Args:
        - data_path (Path): Path to processed loan data.
        - years_quarters_list (list): List of tuples with year and quarter to sample.
        - train_obs_dates (list): Observation dates of training set.
        - test_obs_dates (list): Observation dates of test set.
        - dev_columns (list): List of columns to select.
        - train_size (int): Number of loans to sample for training set.
        - output_path (Path): Path to save the samples.
        - binary_output (bool): Whether to save binary or regular samples.
        - use_economic_data (bool): Whether to include economic features.
        - econ_data_path (Path): Path to the economic data.
    '''

    # Load the developemnt sample
    df_train, df_test = sample_train_test_sets(data_path, 
                                               years_quarters_list, 
                                               train_obs_dates, 
                                               test_obs_dates,
                                               train_size,
                                               test_size,
                                               seed)

    # Process training sample
    df_train = process_sample(df_train, 
                              dev_columns, 
                              categorical_encodings, 
                              num_thresholds,
                              binary_output, 
                              use_economic_data, 
                              econ_data_path,
                              is_train_set=True)
    
    # Process test sample
    df_test = process_sample(df_test, 
                             dev_columns, 
                             categorical_encodings,
                             num_thresholds,
                             binary_output, 
                             use_economic_data, 
                             econ_data_path,
                             is_train_set=False)

    # Saving datasets to parquet
    save_sample_to_disk(df_train, 
                        df_test, 
                        output_path, 
                        binary_output)

    return df_train, df_test
