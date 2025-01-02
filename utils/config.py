from pathlib import Path
from typing import List, Tuple
import yaml
from utils.logging import Colors, log_info


def generate_quarters_list(start_year_quarter: Tuple[int], 
                            end_year_quarter: Tuple[int]) -> List[Tuple[int]]:
    """Generate quarters list from start and end points.
    
    Args:
        start (List[int]): [year, quarter] start point
        end (List[int]): [year, quarter] end point
    
    Returns:
        List[List[int]]: List of [year, quarter] pairs
    """

    years_quarters_list = []
    year, quarter = start_year_quarter

    while (year, quarter) <= end_year_quarter:
        years_quarters_list.append((year, quarter))
        quarter += 1
        if quarter > 4:
            quarter = 1
            year += 1

    return years_quarters_list


def load_config() -> dict:
    """Load and process configuration from config.yaml
    
    Returns:
        dict: Processed configuration with resolved quarters
    """
    config_path = Path("config.yml")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Process years_quarters based on mode
    download_config = config["process"]
    mode = download_config["mode"]
    
    # Create year-quarter list if only start and end quarter are passed
    if mode == "start_end":
        start = tuple(download_config["start_end"]["start"])
        end = tuple(download_config["start_end"]["end"])
        years_quarters_list = generate_quarters_list(start, end)
    elif mode == "list":
        years_quarters_list = download_config["years_quarters_list"]
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'start_end' or 'list'")
    
    # Update config with processed quarters
    config["process"]["years_quarters_list"] = years_quarters_list
    
    return config


def get_missing_years_quarters(config: dict,
                               data_type: str) -> List[Tuple[int]]:
    
    # Get year-quarter list
    years_quarters_list = config['process']['years_quarters_list']
    # Check if data are available for all year-quarter combinations
    missing_years_quarters = []
    # Loop over year-quarter combinations
    for year, quarter in years_quarters_list:
        # Define folder 
        if data_type == "loan_data":
            output_folder = Path(config['paths'][data_type]) / f'historical_data_{year}Q{quarter}.zip'
        elif data_type == "processed_data":
            output_folder = Path(config['paths'][data_type]) / f"{year}Q{quarter}"
        # Check if folder exists, otherwise appned to missing list
        if not output_folder.exists():
            missing_years_quarters.append((year, quarter))
    
    return missing_years_quarters


def get_model_config(model_name):
    """Get model configuration from config file"""
    # Load config
    config = load_config()

    if 'train' not in config:
        raise ValueError("No train section in config")
    if model_name not in config['train']:
        raise ValueError(f"Model {model_name} not found in config")
    
    # Define model configurations
    model_configs = config['train'][model_name]

    # Add data and output paths
    model_configs['data_path'] = Path(config['paths']['dev_sample'])
    model_configs['models_path'] = Path(config['paths']['models_dir'])

    return model_configs