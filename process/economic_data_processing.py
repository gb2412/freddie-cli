from pathlib import Path
import polars as pl


def get_bls_income(data_path: Path):
    '''
    Import and transform BLS median income data.

    Args:
        - data_path (Path): path to the economic data folder.
    '''

    # Read csv into polars dataframe
    BLS_income = pl.read_csv(data_path / 'BLS_Income.csv')

    BLS_income = (BLS_income
                # selecet useful columns
                .select(['Month','State','Median_Ann_Income'])
                # convert month column into last day of the month
                .with_columns(
                    pl.col('Month')
                        .str.strptime(pl.Date, '%Y-%m')
                        .dt.month_end())
                # fill nulls with the last known value partitioning by state
                .group_by('State', maintain_order=True)
                    .map_groups(lambda group_df: group_df.sort('Month').upsample('Month', every='1mo').fill_null(strategy='forward'))
    )

    return BLS_income


def get_bls_unemployment(data_path: Path):
    '''
    Import and transform BLS unemployment data.

    Args:
        - data_path (Path): path to the economic data folder.
    '''

    # Read csv into polars dataframe
    BLS_unemp = pl.read_csv(data_path / 'BLS_Unemp.csv')

    BLS_unemp = (BLS_unemp
                .with_columns([
                    # format zip code to 3 digits
                    pl.col('zip')
                        .cast(str).str.zfill(3).alias('ZIP_Code'),
                    pl.col('unemp_rate').alias('Unemp_Rate'),
                    # convert month column into last day of the month
                    pl.col('Month')
                        .str.strptime(pl.Date, '%Y-%m')
                        .dt.month_end()
                ])
                .select(['Month','ZIP_Code','Unemp_Rate'])
    )

    return BLS_unemp


def get_bls_inflation(data_path: Path):
    '''
    Import and transform BLS inflation data.

    Args:
        - data_path (Path): path to the economic data folder.
    '''

    # Read excel into polars dataframe
    BLS_infl = pl.read_excel(data_path / 'BLS_Infl.xlsx')

    BLS_infl = (BLS_infl
                # filter out semester values
                .filter(~pl.col('Period').is_in(['S01','S02']))
                .with_columns([
                    # combine year and month columns
                    (pl.col('Year').cast(str) + '-' + pl.col('Period').cast(str).str.slice(1)).alias('Month'),
                    pl.col('Value').alias('Infl_Rate')
                ])
                .with_columns([
                    # convert month column into last day of the month
                    pl.col('Month')
                        .str.strptime(pl.Date, '%Y-%m')
                        .dt.month_end()
                ])
                .select(['Month','Infl_Rate'])
    )

    return BLS_infl


def get_house_price_index(data_path: Path):
    '''
    Import and transform FHFA house price index data.

    Args:
        - data_path (Path): path to the economic data folder.
    '''

    # Read excel into polars dataframe
    FHFA_house_price_index = pl.read_excel(data_path / 'FHFA_House_Price_Index.xlsx')

    FHFA_house_price_index = (FHFA_house_price_index
                            .with_columns([
                                # format zip code to 3 digits
                                pl.col('ZIP Code')
                                    .cast(str).str.zfill(3)
                                    .alias('ZIP_Code'),
                                # combine year and quarter columns and get last day of the month
                                (pl.col('Year').cast(str) + '-' + (pl.col('Quarter')*3).cast(str).str.zfill(2))
                                    .str.strptime(pl.Date, '%Y-%m')
                                    .dt.month_end()
                                    .alias('Month'),
                                pl.col('Index').alias('HP_Index')
                            ])
                            .select(['Month','ZIP_Code','HP_Index'])
                            # fill nulls with the last known value partitioning by zip code
                            .group_by('ZIP_Code', maintain_order=True)
                                .map_groups(lambda group_df: group_df.sort('Month').upsample('Month', every='1mo').fill_null(strategy='forward'))
    )

    return FHFA_house_price_index


def get_30yr_frm_rate(data_path: Path):
    '''
    Import and transform FM 30-year fixed rate mortgage average data.

    Args:
        - data_path (Path): path to the economic data folder.
    '''

    # Read excel into polars dataframe
    FM_30yr_FRM_Rate = pl.read_excel(data_path / 'FM_30yr_FRM_Rate.xlsx')

    FM_30yr_FRM_Rate = (FM_30yr_FRM_Rate
                        .with_columns(
                            # turn week column into last day of the month
                            pl.col('Date')
                                .cast(pl.Date)
                                .dt.month_end()
                                .alias('Month'))
                        # get value available on the last day of the month
                        .group_by('Month', maintain_order=True).last()
                        .select(['Month', 'US_30yr_FRM'])
    )

    return FM_30yr_FRM_Rate


def get_merged_econ_data(data_path: Path,
                         loan_data: pl.DataFrame) -> pl.DataFrame:
    '''
    Merge economic data to loans data.

    Args:
        - data_path (Path): path to the economic data folder.
        - loan_data (pl.DataFrame): loans data.
    '''

    # Read csv into polars dataframe
    BLS_income = get_bls_income(data_path)
    BLS_unemp = get_bls_unemployment(data_path)
    BLS_infl = get_bls_inflation(data_path)
    FHFA_house_price_index = get_house_price_index(data_path)
    FM_30yr_FRM_Rate = get_30yr_frm_rate(data_path)

    # Merge all economic data to loans data
    # Offsets are used to match the economic data release calendar
    merged_data = (
        loan_data
            .join(BLS_income, left_on=['Month','Property_State'], right_on=['Month','State'], how='left')
            .join(BLS_unemp.with_columns(pl.col('Month').dt.offset_by('1mo').dt.month_end()), on=['Month','ZIP_Code'], how='left')
            .join(BLS_infl.with_columns(pl.col('Month').dt.offset_by('1mo').dt.month_end()), on='Month', how='left')
            .join(FHFA_house_price_index.with_columns(pl.col('Month').dt.offset_by('2mo').dt.month_end()), on=['Month','ZIP_Code'], how='left')
            .join(FHFA_house_price_index.with_columns(pl.col('Month').dt.offset_by('2mo').dt.month_end()), left_on=['First_Payment_Month','ZIP_Code'], right_on=['Month','ZIP_Code'], how='left',suffix='_Orig')
            .join(FHFA_house_price_index.with_columns(pl.col('Month').dt.offset_by('14mo').dt.month_end()), left_on=['Month','ZIP_Code'], right_on=['Month','ZIP_Code'], how='left',suffix='_year')
            .join(FM_30yr_FRM_Rate, on='Month', how='left')
            .join(FM_30yr_FRM_Rate, left_on='First_Payment_Month', right_on ='Month', how='left', suffix='_Orig')
    )

    return merged_data


def add_comb_features(merged_df: pl.DataFrame) -> pl.DataFrame:
    '''
    Add new features combining loan and c=economic variables.

    Args:
        - merged_df (pl.DataFrame): merged loans and economic data.
    '''

    merged_df = merged_df.with_columns([
        # Difference between the interest rate at origination and the National Mortgage Rate (Freddie Mac) at origination
        (pl.col('Int_Rate') - pl.col('US_30yr_FRM_Orig')).alias('Int_Rate_Diff'),
        
        # Difference between current interest rate and current National Mortgage Rate (Freddie Mac)
        (pl.col('Int_Rate') - pl.col('US_30yr_FRM')).alias('Curr_Int_Rate_Diff')
    ])

    return merged_df


def add_econ_features(merged_data: pl.DataFrame) -> pl.DataFrame:
    '''
    Add new economic features.

    Args:
        - merged_data (pl.DataFrame): merged loans and economic data.
    '''

    merged_data = merged_data.with_columns([
        # Percentage change in the house price index (FHFA) since mortgage origination
        (pl.col('HP_Index') / pl.col('HP_Index_Orig') - 1).alias('HPI_Change_Orig'),
        
        # Percentage change in the house price index (FHFA) in the last 12 months
        (pl.col('HP_Index') / pl.col('HP_Index_year') - 1).alias('HPI_Change_yoy'),
        
        # Log median annual income
        (pl.col('Median_Ann_Income') + 1).log10().alias('Log_Median_Ann_Income')
    ])

    merged_data = add_comb_features(merged_data)

    return merged_data
