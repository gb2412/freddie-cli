import gc
import io
import time

import zipfile
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.logging import Colors, log_info


def setup_directory(path: Path):
    '''
    Create directory if doesn't exists.

    Args:
        - path (Path): Path object.
    '''
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_login_response(username,
                       password,
                       login_page_url,
                       auth_page_url,
                       download_page_url):
    '''
    Login into Freddie Mac website, accept terms and conditions.
    Return session object and response content.

    Args:
        - username (str): Freddie Mac username.
        - password (str): Freddie Mac password.
        - login_page_url (str): URL to login page.
        - auth_page_url (str): URL to authentication page.
        - download_page_url (str): URL to download page.
    '''

    try:
        # Initialize requests session.
        with requests.Session() as sess:
            # Get response from login page.
            response = sess.get(login_page_url)
            # Check if login page loaded successfully
            if response.status_code != 200:
                raise RuntimeError(f"{Colors.FAIL}ERROR{Colors.ENDC}: failed to load login page. Status code: {response.status_code}")

            # Define login payload
            login_payload = {
                'username': username,
                'password': password,
                'pagename': '../menu'
            }
            headers = {
                'User-Agent': 'Chrome/58.0.3029.110'
            }
            # Post login payload to login page
            response_login = sess.post(auth_page_url, data=login_payload, headers=headers)
            log_info(f"Login response status code: {response_login.status_code}")

            # Check if login was successful
            if response_login.status_code == 200 and "Please log in" not in response_login.text:
                log_info(f"{Colors.OKGREEN}SUCCESS{Colors.ENDC}: Login into freddiemac.embs.com successful")
                # Accept terms and conditions
                download_page_payload = {'accept': 'Yes', 'action': 'acceptTandC', 'acceptSubmit': 'Continue'}
                response_download = sess.post(download_page_url, data=download_page_payload, headers=headers)

                # Check if terms and conditions were accepted
                if response_download.status_code == 200:
                    log_info(f"{Colors.OKGREEN}SUCCESS{Colors.ENDC}: Terms and conditions accepted")
                    # Retun session object
                    return sess, response_download.content
                
                else:
                    raise RuntimeError(f"{Colors.FAIL}ERROR{Colors.ENDC}: Failed to accept terms and conditions. Status code: {response_download.status_code}")
            else:
                raise RuntimeError(f"{Colors.FAIL}ERROR{Colors.ENDC}: Failed to login. Status code: {response_login.status_code} \n{response_login.text}")   
    except requests.RequestException as e:
        raise RuntimeError(f"{Colors.FAIL}ERROR{Colors.ENDC}: An error occurred during the login process. Error: {e}")
    return None, None


def download_with_retry(session, 
                        url:str,
                        year:int,
                        quarter:int,
                        data_dir:Path,
                        max_retries:int=3, 
                        backoff_factor:int=1):
    '''
    Download file with retry logic and content validation.
    
    Args:
        - session (requests.Session): requests Session object
        - url (str): URL to download from
        - year (int): Year of data
        - quarter (int): Quarter of data
        - data_dir (Path): Directory to save data
        - max_retries (int): Maximum number of retry attempts
        - backoff_factor (int): Backoff factor for retry delays
    '''
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504, 429]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    
    for attempt in range(max_retries):
        try:
            # Make request
            response = session.get(url)
            response.raise_for_status()
            
            # Get content
            content = response.content
                
            # Verify zip file signature
            if not content.startswith(b'PK\x03\x04'):
                raise zipfile.BadZipFile(f"File signature does not match zip format: {content[:10]}")
            
            # Try opening as zip to validate
            with io.BytesIO(content) as buf:
                with zipfile.ZipFile(buf) as zf:
                    # Verify file listing
                    files = zf.namelist()
                    if not files:
                        raise ValueError("Empty zip file")
                    log_info(f"Found {len(files)} files in {year}Q{quarter} zip archive: {files}")

            # Save zipped data archive
            zip_path = data_dir / f"historical_data_{year}Q{quarter}.zip"
            zip_path.write_bytes(response.content)
            print(f"{Colors.OKGREEN}Successfully saved {zip_path}{Colors.ENDC}")
            return True
        
        except Exception as e:
            if attempt == max_retries - 1:
                # On final attempt, just log error and return
                log_info(f"{Colors.FAIL}Download failed for {year}Q{quarter}{Colors.ENDC}: {str(e)}. \
                         \nPossible causes are unstable connection and corruption during download. \
                         \nRun `freddie download` to retry after all other quarters have been downloaded.")
                return False
            else:
                # Log warning and retry
                log_info(f"{Colors.WARNING}WARNING{Colors.ENDC}: Download attempt {attempt + 1}/{max_retries} for {year}Q{quarter} failed: {str(e)}")
                time.sleep(backoff_factor * (2 ** attempt))


def download_save_data(username,
                      password,
                      years_quarters,
                      data_dir,
                      login_page_url,
                      auth_page_url,
                      download_page_url):
    '''
    Main logic for downloading and saving Freddie Mac data.
    Login and download each year-quarter combination iteratively.

    Args:
        - username (str): Freddie Mac username.
        - password (str): Freddie Mac password.
        - years_quarters (list): list of tuples with year and quarter combinations.
        - data_dir (str): Directory to save data.
        - login_page_url (str): URL to login page.
        - auth_page_url (str): URL to authentication page.
        - download_page_url (str): URL to download page.
    '''

    log_info(f"Starting downloading {len(years_quarters)} year-quarter combinations")

    # Create data directory if not exists
    data_dir = setup_directory(Path(data_dir))

    # Login to Freddie Mac website
    session, _ = get_login_response(username, password, login_page_url, auth_page_url, download_page_url)

    if session is None:
        raise RuntimeError(f"{Colors.FAIL}ERROR{Colors.ENDC}: Failed to establish a session.")

    # Loop over each year-quarter
    for year, quarter in years_quarters:
        # Year-quarter folder URL
        data_url = download_page_url + '?f=historical_data_' + str(year) + 'Q' + str(quarter)
        # Download and save data folder with validation and retry logic
        download_with_retry(session, data_url, year, quarter, data_dir)
        # Call garbage collector to free up memory
        gc.collect()

    
    



