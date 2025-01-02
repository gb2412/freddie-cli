# Freddie CLI
A simple CLI to download, process, and train ML models on the [Freddie Mac's Single Family Loan-Level Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset).

Freddie CLI has been developed during the realization of the following research paper: Interpretable Machine Learning in Credit Risk Modelling. 
The CLI is a handy tool for anyone interacting with Freddie Mac's data for analysis and model development.

Built with [polars](https://pola.rs/) and [click](https://click.palletsprojects.com/en/stable/).

©️ [Giulio Bellini](https://www.linkedin.com/in/giuliobellini/)

## Table of Contents
- Prerequisites
- Setup
- Commands
   - download
   - process
   - sample
   - train
- Configurations
- OOM Errors

## Prerequisites
**[Python](https://www.python.org/downloads/) 3.8 or higher**: To train sparse GAM models with the [fastsparsegams](https://pypi.org/project/fastsparsegams/) library as in the paper, Python 3.8-3.11 is required.

## Setup
1. **Clone the repository**: Open a terminal, navigate to your desired directory, and clone the repository using:
   ```bash
   git clone https://github.com/gb2412/freddie-cli.git
   cd freddie-cli
   ```
2. **Install the CLI**: Install the CLI and all the required Python packages using:
   ```bash
   pip install -e .
   ```

## Commands
Freddie CLI is based on four simple commands that execute Python scripts from the corresponding folders.

### [`download`](download/)
Download Freddie Mac's Single Family Loan-Level Dataset for the quarters specified in the [configurations](#Configurations). 

If a quarter has already been downloaded, it is skipped by default.

| Option         | Short | Description              |
| -------------- | ----- | ------------------------ |
| --refresh | -r    | Re-download all quarters |

```bash
freddie download  # Download missing quarters
freddie download --refresh  # Re-download all quarters
```

### [`process`](process/)
Process raw data to obtain usable datasets for exploratory analysis and model development.

Processing steps:
1. Upload origination and monthly performance data by quarter as a [polars](https://pola.rs/) data frame.
2. Set dataframe schemas.
3. Encode missing values and enums according to the dataset documentation.
4. Create the binary target variable to develop PD models according to the definition of default provided in the paper.
5. Create new features.
6. Save data frames as [parquet](https://parquet.apache.org/) files.

If a quarter has already been processed, it is skipped by default.

| Option         | Short | Description              |
| -------------- | ----- | ------------------------ |
| --refresh | -r    | Re-process all quarters |

```bash
freddie process  # Process missing quarters
freddie process --refresh  # Re-download all quarters
```

### [`sample`](sample/)
Create training and test sets by sampling from the processed data.
Sampling dates and sizes are defined in the [configurations](config.py).

Both standard and binary sets can be created according to the data format required by the model to be trained.

Optionally, the dataset can be enriched with economic data. 
The following economic time series up to June 2024 are available under [data/economic](data/economic/).

| Name           | Schema | Description              | Source |
| -------------- | -----  | ------------------------ | ------ |
| BLS_Income.csv| Month, State, Median_Ann_Income| Median annual income by state | [BLS OEWS](https://www.bls.gov/oes/) |
| BLS_Unemp.csv |Month, zip, unemp_rate | Unemployment Rate by county | [BLS LAUS](https://www.bls.gov/lau) |
| BLS_Infl.xlsx | year, period, value | National inflation rate | [BLS CPI](https://www.bls.gov/cpi) |
| FHFA_House_Price_Index.xlsx | ZIP Code, Year, Quarter, Index | House price index by county | [FHFA HPI](https://www.fhfa.gov/data/hpi) |
| FM_30yr_FRM_Rate.xlsx | Date, US_30yr_FRM | US 30-year fixed rate mortgage average | [FREDDIE MAC](https://www.freddiemac.com/pmms) |

To add new economic variables or change the schema or format of existing ones, 
the economic data processing [script](process/economic_data_processing.py) must be adapted.

Sampling steps:
1. Upload relevant origination quarters and observation dates as a [polars](https://pola.rs/) data frame.
2. Filter out loans in default at the time of observation.
3. Add economic variables, if required.
4. Select relevant features.
5. Split training and test sets.
6. If binary output required:
   7. Encode missing values as binary features.
   8. Encode categorical variables: one binary feature for each value.
   9. Encode continuous variables: create binary variables of type `Var_Name <= threshold` for each percentile multiple of 5 (20 thresholds in total).
10. Save training and test sets as [parquet](https://parquet.apache.org/)

| Option         | Short | Description              |
| -------------- | ----- | ------------------------ |
| --refresh | -r | Re-sample training and test set |
| --use-econ-data | -eco | Include economic data |
| --binary | -bin | Create binary training and test sets |

```bash
freddie sample  # Process training and test sets
freddie sample --refresh  # Re-sample sets
freddie sample --use-econ-data # Include economic data
freddie sample --binary # Create binary sets
```

### [`train`](train/)
Train and save custom models on the sampled training set.

The repo contains training scripts for sparse GAM, Random Forest, and XGBoost. 
Custom training scripts can be created in the following steps:
1. Define the model name and parameters in the configurations [file](config.py)
2. Create a script in the [train](train/) as `<model_name>.py`. Use the same name as in the configuration.
3. The script should import the training set, fit the model, and save it. Use the existing scripts as templates. 

| Option         | Short | Description              |
| -------------- | ----- | ------------------------ |
| --model | -m | Pass the name of the model to be trained |

```bash
freddie train --model 'model_name'  # Process training and test sets
```

## Configurations
The [configuration file](config.yml) allows customization of the resulting datasets and models without directly editing the code.
The file contains the configurations used in the paper but can be easily edited to suit the user's purposes and resources.

Below is a detailed description of all available configuration settings.

#### [Paths Configuration](config.yml#L1)
| Field | Description |
|-------|-------------|
| `paths.loan_data` | Directory for raw Freddie Mac loan data files |
| `paths.processed_data` | Output directory for processed loan data |
| `paths.economic_data` | Directory containing economic data |
| `paths.dev_sample` | Output directory for model development datasets |
| `paths.models_dir` | Directory where trained models are saved |

#### Download Settings
| Field | Description |
|-------|-------------|
| `download.login_page_url` | Freddie Mac login page URL |
| `download.auth_page_url` | Authentication endpoint URL |
| `download.download_page_url` | Data download page URL |

#### Processing Settings
| Field | Description |
|-------|-------------|
| `process.mode` | Data quarters selection mode: "list" or "start_end" |
| `process.start_end.start` | Starting year and quarter [YYYY, Q] when mode="start_end" |
| `process.start_end.end` | Ending year and quarter [YYYY, Q] when mode="start_end" |
| `process.years_quarters_list` | List of [year, quarter] pairs when mode="list" |
| `process.batch_size` | Number of loans to process in each batch |

#### Sampling Settings
| Field | Description |
|-------|-------------|
| `sample.include_economic_data` | Whether to include economic variables |
| `sample.train_obs_dates` | List of observation dates "YYYY-MM" for training set |
| `sample.test_obs_dates` | List of observation dates "YYYY-MM" for test set |
| `sample.train_size` | Training set size |
| `sample.dev_columns.mortgage_columns` | List of loan-level features to include |
| `sample.dev_columns.economic_columns` | List of economic features to include |

#### Model Training Settings
| Field | Description |
|-------|-------------|
| `train.<model_name>` | Name of the model |
| `train.<model_name>.<parameter_name>` | Model parameter |








