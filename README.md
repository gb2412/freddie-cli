# Freddie CLI
A simple CLI to download, process, and train ML models on the [Freddie Mac's Single Family Loan-Level Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset).

Freddie CLI has been developed during the realization of the following research paper: Interpretable Machine Learning in Credit Risk Modelling. 
The CLI is a very useful tool for anyone interacting with Freddie Mac's data for analysis and model development.

Built with [polars](https://pola.rs/) and [click](https://click.palletsprojects.com/en/stable/).

Author: [Giulio Bellini](https://www.linkedin.com/in/giuliobellini/)

## Prerequisites
**[Python](https://www.python.org/downloads/) 3.8 or higher**: To train sparse GAM models with the fastsparsegams library as in the paper, Python 3.8-3.11 is required.

## Setup
1. **Clone the repository**: Open a terminal, navigate to your desired directory, and colne the repository using:
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
Download Freddie Mac's Single Family Loan-Level Dataset for the quarters specified in [`config.py`](config.py). 

If a quarter has already been downloaded, it is skipped by default.

| Option         | Short | Description              |
| -------------- | ----- | ------------------------ |
| --refresh | -r    | Re-download all quarters |

```bash
freddie download  # Download missing quarters
freddie download --refresh  # Re-download all quarters
```

### [`process`](process/)
Process the dataset by:
1. Uploading origination and monthly performance data by quarter as a polars dataframe.
2. Setting dataframe schemas.
3. Encoding missing values and enums according to the dataset documentation.
4. Creating the binary target variable to develop PD models according to the definition of default defined in the paper.
5. Creating new features.
6. Saving dataframes as parquet files.

If a quarter has already been processed, it is skipped by default.

| Option         | Short | Description              |
| -------------- | ----- | ------------------------ |
| --refresh | -r    | Re-process all quarters |

```bash
freddie process  # Process missing quarters
freddie process --refresh  # Re-download all quarters
```

### [`sample`](sample/)









