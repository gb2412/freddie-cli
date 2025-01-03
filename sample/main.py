import gc
import sys
from pathlib import Path

from utils.logging import Colors, log_info
from utils.config import load_config, get_missing_years_quarters
from sample.sample_selection import process_and_save_train_test


def main(refresh:bool, 
         use_econ_data:bool, 
         binary:bool):
    '''
    Sample Freddie Mac and economic data ans split train and test sets.

    Args:
        - refresh (bool): If True, resample train and test sets. If False, keep existing sets.
        - use_econ_data (bool): If True, include economic data in the sample.
        - binary (bool): If True, convert features to binary.
    '''

    # Load configurations
    config = load_config()

    # Check if data are available for all year-quarter combinations
    missing_years_quarters = get_missing_years_quarters(config, data_type='processed_data')
    # Log missing data
    if missing_years_quarters:
        raise RuntimeError(f"""{Colors.FAIL}ERROR{Colors.ENDC}: Missing data for {len(missing_years_quarters)} \
            \nyear-quarter combinations: {missing_years_quarters} \
            \nTo process the missing data, please run the following command: `freddie process` \
            \nTo process all data, please run the following command: `freddie process --refresh`
            """)

    # Check if refresh is called
    if not refresh:
        # Define train and test paths
        train_path = Path(config['paths']['dev_sample']) / 'train_binary.parquet' if binary else 'train.parquet'
        test_path = Path(config['paths']['dev_sample']) / 'test_binary.parquet' if binary else 'test.parquet'
        # Check if train and test sets already exist
        if train_path.exists() and test_path.exists():
            print(f"Train and test {'binary' if binary else ''} sets already exist. Use --refresh to resample.")
            return

    # Process and save train and test data
    process_and_save_train_test(data_path=Path(config['paths']['processed_data']),
                                years_quarters_list=config['process']['years_quarters_list'],
                                train_obs_dates=config['sample']['train_obs_dates'],
                                test_obs_dates=config['sample']['test_obs_dates'],
                                dev_columns=config['sample']['dev_columns']['mortgage_columns'] + \
                                    config['sample']['dev_columns']['economic_columns'] \
                                    if use_econ_data else config['sample']['dev_columns']['mortgage_columns'],
                                train_size=config['sample']['train_size'],
                                categorical_encodings=config['sample']['features_encodings'],
                                output_path=Path(config['paths']['dev_sample']),
                                binary_output=binary,
                                use_economic_data=use_econ_data,
                                econ_data_path=Path(config['paths']['economic_data']))

    # Call garbage collector to free up memory
    gc.collect()
    
    log_info("Sampling completed")


if __name__ == "__main__":
    # Check if flags are called
    refresh = '--refresh' in sys.argv
    use_econ_data = '--use-econ-data' in sys.argv
    binary = '--binary' in sys.argv
    main(refresh, use_econ_data, binary)