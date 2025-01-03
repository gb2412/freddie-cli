import gc
import sys
from pathlib import Path

from utils.logging import Colors, log_info
from utils.config import load_config, get_missing_years_quarters
from process.mortgage_data_processing import process_year_quarter_in_batches
from sample.sample_selection import process_and_save_train_test


def main(refresh):
    '''
    Process Freddie Mac data.

    Args:
        refresh (bool): If True, re-process all data. If False, process only missing data.
    '''

    # Load configurations
    config = load_config()

    # Check if raw data are available for all year-quarter combinations
    missing_years_quarters = get_missing_years_quarters(config, data_type='loan_data')
    # Log missing data
    if missing_years_quarters:
        raise RuntimeError(f"""{Colors.FAIL}ERROR{Colors.ENDC}: Missing data for {len(missing_years_quarters)} 
            year-quarter combinations: {missing_years_quarters}
            To download the missing data, please run the following command: `freddie download`
            To download all data, please run the following command: `freddie download --refresh`
            """)
    
    # Check if refresh is called
    if not refresh:
        # Get missing year-quarter combinations in processed data
        years_quarters_list = get_missing_years_quarters(config, data_type='processed_data')

        if not years_quarters_list:
            print("All data files already exist. Use --refresh to re-process.")
            return
    else:
        # Get entire list of year-quarter to process
        years_quarters_list = config["process"]["years_quarters_list"]

    log_info(f"Starting processing for {len(years_quarters_list)} year-quarter combinations: {years_quarters_list}")

    # Loop over each year-quarter
    for year, quarter in years_quarters_list:

        # Process data
        process_year_quarter_in_batches(year, 
                                        quarter,
                                        batch_size=config['process']['batch_size'],
                                        data_path=Path(config['paths']['loan_data']),
                                        output_path=Path(config['paths']['processed_data']))
    
        # Call garbage collector to free up memory
        gc.collect()
    
    log_info("All data processing completed")


if __name__ == "__main__":
    # Check if refresh is called
    refresh = '--refresh' in sys.argv
    main(refresh)