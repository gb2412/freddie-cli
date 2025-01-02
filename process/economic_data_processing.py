from pathlib import Path
import polars as pl


def get_bls_income(data_path: Path):

    BLS_income = pl.read_csv(data_path / 'BLS_Income.csv')

    BLS_income = (BLS_income
                .select(['Month','State','Median_Ann_Income'])
                .with_columns(
                    pl.col('Month')
                        .str.strptime(pl.Date, '%Y-%m')
                        .dt.month_end())
                .group_by('State', maintain_order=True)
                    .map_groups(lambda group_df: group_df.sort('Month').fill_null(strategy='forward'))
    )

    return BLS_income


def get_bls_unemployment(data_path: Path):

    BLS_unemp = pl.read_csv(data_path / 'BLS_Unemp.csv')

    BLS_unemp = (BLS_unemp
                .with_columns([
                    pl.col('zip')
                        .cast(str).str.zfill(3).alias('ZIP_Code'), 
                    pl.col('unemp_rate').alias('Unemp_Rate'),
                    pl.col('Month')
                        .str.strptime(pl.Date, '%Y-%m')
                        .dt.month_end()
                ])
                .select(['Month','ZIP_Code','Unemp_Rate'])
    )

    return BLS_unemp


def get_bls_inflation(data_path: Path):

    BLS_infl = pl.read_excel(data_path / 'BRL_Infl.xlsx')

    BLS_infl = (BLS_infl
                .filter(~pl.col('Period').is_in(['S01','S02']))
                .with_columns([
                    (pl.col('Year').cast(str) + '-' + pl.col('Period').cast(str).str.slice(1)).alias('Month'),
                    pl.col('Value').alias('Infl_Rate')
                ])
                .with_columns([
                    pl.col('Month')
                        .str.strptime(pl.Date, '%Y-%m')
                        .dt.month_end()
                ])
                .select(['Month','Infl_Rate'])
    )

    return BLS_infl


def get_home_price_index(data_path: Path):

    FHA_home_price_index = pl.read_excel(data_path / 'FHA_Home_Price_index.xlsx')

    FHA_home_price_index = (FHA_home_price_index
                            .with_columns([
                                pl.col('ZIP Code')
                                    .cast(str).str.zfill(3)
                                    .alias('ZIP_Code'),
                                (pl.col('Year').cast(str) + '-' + (pl.col('Quarter')*3).cast(str).str.zfill(2))
                                    .str.strptime(pl.Date, '%Y-%m')
                                    .dt.month_end()
                                    .alias('Month'),
                                pl.col('Index').alias('HP_Index')
                            ])
                            .select(['Month','ZIP_Code','HP_Index'])
                            .group_by('ZIP_Code', maintain_order=True)
                                .map_groups(lambda group_df: group_df.sort('Month').fill_null(strategy='forward'))
    )

    return FHA_home_price_index


def get_30yr_frm_rate(data_path: Path):

    FM_30yr_FRM_Rate = pl.read_excel(data_path / 'FM_30yr_FRM_Rate.xlsx')

    FM_30yr_FRM_Rate = (FM_30yr_FRM_Rate
                        .with_columns(
                            pl.col('Date')
                                .cast(pl.Date)
                                .dt.month_end()
                                .alias('Month'))
                        .group_by('Month', maintain_order=True).last()
                        .select(['Month', 'US_30yr_FRM'])
    )

    return FM_30yr_FRM_Rate


def get_merged_econ_data(data_path: Path,
                         loan_data: pl.DataFrame) -> pl.DataFrame:

    BLS_income = get_bls_income(data_path)
    BLS_unemp = get_bls_unemployment(data_path)
    BLS_infl = get_bls_inflation(data_path)
    FHA_home_price_index = get_home_price_index(data_path)
    FM_30yr_FRM_Rate = get_30yr_frm_rate(data_path)


    merged_data = (
        loan_data
            .join(BLS_income, left_on=['Month','Property_State'], right_on=['Month','State'], how='left')
            .join(BLS_unemp.with_columns(pl.col('Month').dt.offset_by('1mo')), on=['Month','ZIP_Code'], how='left')
            .join(BLS_infl.with_columns(pl.col('Month').dt.offset_by('1mo')), on='Month', how='left')
            .join(FHA_home_price_index.with_columns(pl.col('Month').dt.offset_by('2mo')), on=['Month','ZIP_Code'], how='left')
            .join(FHA_home_price_index.with_columns(pl.col('Month').dt.offset_by('2mo')), left_on=['First_Payment_Month','ZIP_Code'], right_on=['Month','ZIP_Code'], how='left',suffix='_Orig')
            .join(FHA_home_price_index.with_columns(pl.col('Month').dt.offset_by('14mo')), left_on=['Month','ZIP_Code'], right_on=['Month','ZIP_Code'], how='left',suffix='_year')
            .join(FM_30yr_FRM_Rate, on='Month', how='left')
            .join(FM_30yr_FRM_Rate, left_on='First_Payment_Month', right_on ='Month', how='left', suffix='_Orig')
    )

    return merged_data


def add_comb_features(merged_df: pl.DataFrame) -> pl.DataFrame:

    merged_df = merged_df.with_columns([
        # Difference between the interest rate at origination and the National Mortgage Rate (Freddie Mac) at origination
        (pl.col('Int_Rate') - pl.col('US_30yr_FRM_Orig')).alias('Int_Rate_Diff'),
        
        # Difference between current interest rate and current National Mortgage Rate (Freddie Mac)
        (pl.col('Int_Rate') - pl.col('US_30yr_FRM')).alias('Curr_Int_Rate_Diff')
    ])

    return merged_df


def add_econ_features(merged_data: pl.DataFrame) -> pl.DataFrame:

    merged_data = merged_data.with_columns([
        # Percentage change in the home price index (FHA) since mortgage origination
        (pl.col('HP_Index') / pl.col('HP_Index_Orig') - 1).alias('HPI_Change_Orig'),
        
        # Percentage change in the home price index (FHA) in the last 12 months
        (pl.col('HP_Index') / pl.col('HP_Index_year') - 1).alias('HPI_Change_yoy'),
        
        # Log median annual income
        pl.col('Median_Ann_Income').log10().alias('Log_Median_Ann_Income')
    ])

    merged_data = add_comb_features(merged_data)

    return merged_data
