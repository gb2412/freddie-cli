import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from utils.logging import Colors, log_info
from utils.config import load_config, get_missing_years_quarters
from download.download_freddie_mac_data import download_save_data


def main(refresh):
    """
    Download Freddie Mac data.
    """

    # Load environment variables from .env file
    load_dotenv()
    # Get Freddie Mac username and password
    FM_USERNAME = os.getenv('FM_USERNAME')
    FM_PASSWORD = os.getenv('FM_PASSWORD')

    if not FM_USERNAME or not FM_PASSWORD:
        raise RuntimeError(f"{Colors.FAIL}ERROR{Colors.ENDC}: Freddie Mac username or password not found in environment variables.")
    
    # Load configurations
    config = load_config()

    if not refresh:
        # Filter out existing year-quarter combinations
        years_quarters_list = get_missing_years_quarters(config, data_type="loan_data")
        
        if not years_quarters_list:
            log_info("All data files already exist. Use --refresh to re-download.")
            return
    else:
        # Get all year-quarter combinations
        years_quarters_list = config["process"]["years_quarters_list"]

    # Download and save Freddie Mac Single-Family Loan-Level Dataset
    download_save_data(username=FM_USERNAME,
                      password=FM_PASSWORD,
                      years_quarters=years_quarters_list,
                      data_dir=config["paths"]["loan_data"],
                      login_page_url=config["download"]["login_page_url"],
                      auth_page_url=config["download"]["auth_page_url"],
                      download_page_url=config["download"]["download_page_url"])


if __name__ == '__main__':
    refresh = '--refresh' in sys.argv
    main(refresh)