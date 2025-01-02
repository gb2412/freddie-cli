from zipfile import ZipFile
from pathlib import Path
from io import BytesIO

import polars as pl

from utils.logging import log_memory_usage, Colors, log_info


def load_origination_data(year, 
                          quarter,
                          data_path: Path):
    '''
    Import origination data for a given year and quarter from the Freddie Mac dataset.
    Unzip data and read the text file into a polars DataFrame enforcing the schema.

    Parameters:
    - year (int): The year of the data to import.
    - quarter (int): The quarter of the data to import.
    '''

    # Path to zip folder
    zip_folder_path = data_path / f'historical_data_{year}Q{quarter}.zip'
    # Open zip folder
    zip_folder = ZipFile(zip_folder_path)
    # Read zip content into memory buffer
    orig_buffer = BytesIO(zip_folder.read(f'historical_data_{year}Q{quarter}.txt'))

    # Define origination data schema
    origination_schema = {
        "Credit_Score": pl.Int32,  # The credit score of the borrower (300-850).
        "First_Payment_Month": pl.Utf8,  # The month of the first payment for the loan.
        "First_Time_Home_Buyer_Flag": pl.Utf8,  # Indicates if the borrower is a first-time home buyer.
        "Maturity_Date": pl.Utf8,  # The date when the loan matures.
        "Metropolitan_Area": pl.Utf8,  # The metropolitan area of the property securing the loan.
        "MI_Percentage": pl.Float64,  # The mortgage insurance percentage of the loan.
        "Num_Units": pl.Int32,  # The number of units in the property securing the loan.
        "Occupancy_Status": pl.Utf8,  # The occupancy status of the property securing the loan.
        "CLTV": pl.Float64,  # The combined loan-to-value ratio of the loan.
        "DTI": pl.Float64,  # The debt-to-income ratio of the borrower.
        "UPB": pl.Float64,  # The unpaid principal balance of the loan.
        "LTV": pl.Float64,  # The loan-to-value ratio of the loan.
        "Int_Rate": pl.Float64,  # The interest rate of the loan.
        "Channel": pl.Utf8,  # The origination channel of the loan.
        "Prepay_Penalty": pl.Utf8,  # Indicates if there is a prepayment penalty on the loan.
        "Amortization_Type": pl.Utf8,  # The type of amortization for the loan.
        "Property_State": pl.Utf8,  # The state where the property securing the loan is located.
        "Property_Type": pl.Utf8,  # The type of property securing the loan.
        "ZIP_Code": pl.Utf8,  # The three-digit ZIP code of the property securing the loan.
        "Loan_Num": pl.Utf8,  # Unique identifier for each loan.
        "Loan_Purpose": pl.Utf8,  # The purpose of the loan (e.g., purchase, refinance).
        "Loan_Term": pl.Int32,  # The term of the loan in months.
        "Num_Borrowers": pl.Int32,  # The number of borrowers on the loan.
        "Seller_Name": pl.Utf8,  # The name of the seller of the loan.
        "Servicer_Name": pl.Utf8,  # The name of the servicer of the loan.
        "Sup_Conforming_Flag": pl.Utf8,  # Indicates if the loan is a super conforming loan.
        "Pre_Ref_Loan_Num": pl.Utf8,  # The pre-relief refinance loan number.
        "Program_Indicator": pl.Utf8,  # The program indicator for the loan.
        "Ref_Indicator": pl.Utf8,  # The relief refinance indicator for the loan.
        "Property_Val_Meth": pl.Utf8,  # The method used to value the property securing the loan.
        "Int_Only_Indicator": pl.Utf8,  # Indicates if the loan is interest-only.
        "MI_Canc_Indicator": pl.Utf8  # Indicates if the mortgage insurance was canceled.
    }

    # Import origination data
    origination_data = pl.read_csv(orig_buffer, 
                                   schema=origination_schema, 
                                   has_header = False,
                                   separator='|') # type: ignore

    return origination_data


def load_performance_data(year, 
                          quarter,
                          data_path: Path,
                          loan_ids):
    '''
    Import performance data for a given year and quarter from the Freddie Mac dataset.
    Unzip data and read the text file into a polars DataFrame enforcing the schema.

    Parameters:
    - year (int): The year of the data to import.
    - quarter (int): The quarter of the data to import.
    '''

    # Path to zip folder
    zip_folder_path = data_path / f'historical_data_{year}Q{quarter}.zip'
    zip_folder = ZipFile(zip_folder_path)
    # Read zip content into memory buffer
    perf_buffer = BytesIO(zip_folder.read(f'historical_data_time_{year}Q{quarter}.txt'))
    
    # Define performance data schema
    performance_schema = {
        "Loan_Num": pl.Utf8,  # Unique identifier for each loan.
        "Month": pl.Utf8,  # The month the performance data was reported.
        "Current_Actual_UPB": pl.Float64,  # The current unpaid principal balance of the loan.
        "Current_Delinquency_Status": pl.Utf8,  # The current delinquency status of the loan.
        "Loan_Age": pl.Int32,  # The age of the loan in months.
        "Months_to_Maturity": pl.Int32,  # The number of months remaining until the loan matures.
        "Defect_Settlement_Date": pl.Utf8,  # The date of the defect settlement.
        "Modification_Flag": pl.Utf8,  # Indicates if the loan has been modified in current or prior period.
        "Zero_Balance_Code": pl.Utf8,  # The code indicating the reason for a zero balance.
        "Zero_Balance_Effective_Date": pl.Utf8,  # The effective date of the zero balance.
        "Current_Int_Rate": pl.Float64,  # The current interest rate of the loan.
        "Current_Non_Int_UPB": pl.Float64,  # The current non-interest bearing unpaid principal balance.
        "DDLPI": pl.Utf8,  # Due date of last paid installment.
        "MI_Recoveries": pl.Float64,  # The mortgage insurance recoveries.
        "Net_Sale_Proceeds": pl.Float64,  # The net proceeds from the sale of the property.
        "Non_MI_Recoveries": pl.Float64,  # The non-mortgage insurance recoveries.
        "Total_Expenses": pl.Float64,  # The total expenses related to acquiring, maintaining and/or disposing the property.
        "Legal_Costs": pl.Float64,  # The legal costs associated with the sale of the property.
        "Maintenance_And_Preservation_Costs": pl.Float64,  # The maintenance and preservation costs associated with the sale of the property.
        "Taxes_And_Insurance": pl.Float64,  # The taxes and insurance associated with the sale of the property.
        "Miscellaneous_Expenses": pl.Float64,  # The miscellaneous expenses associated with the sale of the property.
        "Actual_Loss": pl.Float64,  # The actual loss on the loan.
        "Cum_Modification_Cost": pl.Float64,  # The cumulative modification cost.
        "Step_Modification_Flag": pl.Utf8,  # Indicates if the loan has a step modification.
        "Payment_Deferral_Flag": pl.Utf8,  # Indicates if the payment has been deferred.
        "ELTV": pl.Float64,  # The estimated current loan-to-value ratio of the loan.
        "Zero_Bal_Removal_UPB": pl.Float64,  # The unpaid principal balance at the time of zero balance removal.
        "Delinquent_Accrued_Int": pl.Float64,  # The delinquent accrued interest owned at default.
        "Delinquency_Disaster_Flag": pl.Utf8,  # Indicates if the loan is delinquent due to a disaster.
        "Borrower_Assistance_Status_Code": pl.Utf8,  # The assistance plan the borrower is enrolled in.
        "Curr_Month_Modification_Cost": pl.Float64,  # The modification cost for the current month.
        "Curr_Int_UPB": pl.Float64  # The current interest bearing unpaid principal balance.
    }
    
    # Read the CSV batch
    perf_data = (pl.read_csv(
                    perf_buffer,
                    schema=performance_schema,
                    has_header=False,
                    separator='|')
                    .filter(pl.col('Loan_Num').is_in(loan_ids))
    )
    
    return perf_data


def change_column_encoding(orig_data, perf_data):
    '''
    Change column format and econding.
    Standardize missing values specification.

    Parameters:
        - orig_data (polars.DataFrame): The origination data.
        - perf_data (polars.DataFrame): The performance data.
    '''

    # Convert to datetime
    orig_data = orig_data.with_columns(
        pl.col("First_Payment_Month")
        .str.strptime(pl.Date, format="%Y%m")
        .dt.month_end()
        )
    
    # Convert to datetime
    perf_data = perf_data.with_columns(
        pl.col("Month")
            .str.strptime(pl.Date, format="%Y%m")
            .dt.month_end()
        )

    # Define missing value encodings for each column in origination data
    orig_missing_map = {
        'Credit_Score': 9999,
        'MI_Percentage': 999,
        'Num_Units': 99,
        'CLTV': 999,
        'DTI': 999,
        'LTV': 999,
        'Num_Borrowers': 99,
        'First_Time_Home_Buyer_Flag':'9',
        'Occupancy_Status':'9',
        'Channel':'9',
        'Property_Type':'99',
        'ZIP_Code':'00',
        'Loan_Purpose':'9',
        'Program_Indicator':'9',
        'Property_Val_Meth':'9',
        'MI_Canc_Indicator':'9',
        'MI_Canc_Indicator':'7'
    }

    # Handle missing values in origination data
    orig_miss_exprs = [
        pl.when(pl.col(col) == val)
                    .then(None)
                    .otherwise(pl.col(col))
                    .alias(col)
        for col, val in orig_missing_map.items()
    ]
    # Apply transformations
    orig_data = orig_data.with_columns(
                orig_miss_exprs + [
                pl.col("Sup_Conforming_Flag").fill_null('N'),
                pl.col("Ref_Indicator").fill_null('N')
    ])

    # Handle missing values in performance data
    perf_data = perf_data.with_columns([
        pl.when(pl.col('ELTV') == 999)
                    .then(None)
                    .otherwise(pl.col('ELTV')),
        pl.col('Delinquency_Disaster_Flag').fill_null('N'),
        pl.col("Modification_Flag").fill_null('N')
    ])

    # Change encoding in performance data
    orig_data = orig_data.with_columns([
        pl.col('Property_Val_Meth').replace({
                '2': 'A',
                '3': 'O', 
                '1': 'AC',
                '4': 'ACP'
            })
    ])

    return orig_data, perf_data


def add_loan_target_and_features(orig_data,
                                perf_data, 
                                year, 
                                quarter):
    '''
    Apply transformations to the origination and performance data.

    Parameters:
    - orig_data (polars.DataFrame): The origination data.
    - perf_data (polars.DataFrame): The performance data.
    - year (int): The year of the data.
    - quarter (int): The quarter of the data.
    '''
    
    # Filter out defected loans from origination data
    orig_data = (orig_data
                        .filter(~pl.col("Loan_Num").is_in(perf_data.filter(pl.col("Zero_Balance_Code") == "96")
                            .select("Loan_Num").unique()))
                        )
    
    # Filter out defected loans from performance data
    perf_data = perf_data.join(orig_data.select("Loan_Num").unique(), on="Loan_Num", how="inner")

    # Add year and quarter columns
    # to origination data
    orig_data = orig_data.with_columns([
        pl.lit(year).alias("Year"),
        pl.lit(quarter).alias("Quarter")
    ])
    # to performance data
    perf_data = perf_data.with_columns([
        pl.lit(year).alias("Year"),
        pl.lit(quarter).alias("Quarter")
    ])

    # Add a column to performance data to indicate if the loan is in default
    perf_data = perf_data.with_columns([
            (
                ((pl.col("Current_Delinquency_Status") >= '3').fill_null(False) |
                (pl.col("Zero_Balance_Code").is_in(["02", "03", "09", "15"])).fill_null(False))
            )
            .cast(pl.Int32)
            .alias("Current_Default")
    ])
    # Sort the performance data by Loan_Num and Month
    perf_data = perf_data.sort(by=["Loan_Num", "Month"], descending=[False, True])
    # Add a column to performance data to indicate if the loan is in default at any time
    # during the 12 months following the current month (ever-bad one-year default)
    perf_data = perf_data.with_columns([
        pl.col('Current_Default')
            .rolling_max(window_size=13, weights =  [1]*12 + [0])
            .over("Loan_Num")
            .alias(
                "Default"
                )
    ])
    # Remove the first 12 months fro each loan
    perf_data = perf_data.filter(pl.col("Default").is_not_null())

    # Add columns to represent the performance of the loan over the past 12 months
    perf_data = perf_data.with_columns([

        pl.when(pl.col("Current_Delinquency_Status") == "RA")
                .then(999)
                .otherwise(pl.col("Current_Delinquency_Status")).cast(pl.Int32)
            .rolling_max_by('Month', window_size='12mo')
            .over("Loan_Num")
            .map_elements(lambda x: "RA" if x == 999 else str(x), return_dtype=str)
            .alias(
                "Max_Delinq_Severity_yoy"
                ),

        pl.col("Modification_Flag")
            .eq('Y').cast(pl.Int32)
            .rolling_sum_by('Month', window_size='12mo')
            .over("Loan_Num")
            .alias(
                "Num_Modifications_yoy"
                ),

        pl.col("Current_Delinquency_Status")
            .eq("0").cast(pl.Int32)
            .rolling_sum_by('Month', window_size='12mo')
            .over("Loan_Num")
            .alias(
                "Delinq_Status_0_Count"
                ),

        pl.col("Current_Delinquency_Status")
            .eq("1").cast(pl.Int32)
            .rolling_sum_by('Month', window_size='12mo')
            .over("Loan_Num")
            .alias(
                "Delinq_Status_1_Count"
                ),

        pl.col("Current_Delinquency_Status")
            .eq("2").cast(pl.Int32)
            .rolling_sum_by('Month', window_size='12mo')
            .over("Loan_Num")
            .alias(
                "Delinq_Status_2_Count"
                )
    ])

    # Merge origination and performance data
    merged_data = orig_data.join(perf_data, on="Loan_Num", how="right")

    # Add features to the merged data
    merged_data = merged_data.with_columns([

        # Proportion of the original UPB that has already been repaid
        ((pl.col('UPB') - pl.col('Current_Actual_UPB')) / pl.col('UPB')).alias('Payment_Ratio'),
        
        # Approximation of the principal obligation per month until maturity
        (pl.col('Current_Actual_UPB') / pl.col('Months_to_Maturity')).alias('Balance_Per_Month'),
        
        # Maturity Progress
        (pl.col('Loan_Age') / pl.col('Loan_Term')).alias('Maturity_Progress'),

        # Log monetary quantities
        pl.col('UPB').add(1).log10().alias('Log_UPB'),

        pl.col('Current_Actual_UPB').add(1).log10().alias('Log_Curr_UPB'),

        pl.col('Current_Non_Int_UPB').add(1).log10().alias('Log_Non_Int_Bear_UPB'),

        # Lagged default rate at the ZIP code level
        pl.col('Default')
            .rolling_mean_by('Month', window_size='12mo')
            .over('ZIP_Code')
            .alias(
                'ZIP_Default_Rate'
                )
    ])

    # Convert target column to -1 and 1
    merged_data = merged_data.with_columns([
        pl.col('Default').replace({
            0: -1,
            1: 1
        }).alias('Default')
    ])
    
    return merged_data


def save_batch_to_disk(batch: pl.DataFrame,
                       batch_num: int,
                       output_path: Path):
    '''
    Save a processed batch to disk.
    '''

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the batch to disk
    batch.write_parquet(output_path / f'batch_{batch_num}.parquet')


def process_year_quarter_in_batches(year, 
                                    quarter,
                                    batch_size,
                                    data_path: Path,
                                    output_path: Path):
    '''
    Main processing function: loads origination and performance data, processes it, and saves it to disk.

    Parameters:
    - year (int): The year of the data to import.
    - quarter (int): The quarter of the data to import.
    - output_path (str): The path to save the processed data.
    '''
    
    log_info(f"Processing data for {year}Q{quarter}")

    # Monitor memory consumption
    log_memory_usage()

    # Import origination data
    orig_data = load_origination_data(year, quarter, data_path)

    # Process data in batches
    for i in range(0, len(orig_data), batch_size):

        # Select loans batch
        orig_batch = orig_data.slice(i, batch_size)

        # Import performance data
        perf_batch = load_performance_data(year, quarter, data_path, loan_ids=orig_batch.select('Loan_Num'))

        # Process batch
        # Change encodings
        orig_df, perf_df = change_column_encoding(orig_batch, perf_batch)

        # Add features and target variable
        merged_df = add_loan_target_and_features(orig_df, perf_df, year, quarter)

        # Save merged data
        save_batch_to_disk(merged_df, i//batch_size, output_path / f'{year}Q{quarter}')

        log_info(f"Completed processing batch n. {i//batch_size} for {year}Q{quarter}")

    log_info(f"{Colors.OKGREEN}Completed processing data for {year}Q{quarter}{Colors.ENDC}")