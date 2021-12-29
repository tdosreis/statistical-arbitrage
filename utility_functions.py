import numpy as np
import yfinance as yf
import datetime as dt


def eval_division(val, m=5): 
    if val%m == 0:
        return True
    else:
        return False

    
def mask_idx(idx, m=5): 
    nulls = (
        np.array(
            [eval_division(x, m=m) for _, x in enumerate(idx)],
            dtype=bool
        )
    )
    return nulls


def gather_data(tickers, n_years=1):

    n_days = int(365 * n_years)

    assets = {}
    for ticker in tickers:
        print(f'Gathering data for {ticker} for the past {n_years} years')
        data = (
            yf.download(
                tickers=ticker,
                start=dt.date.today() - dt.timedelta(n_days),
                end=dt.datetime.today()
            )
        )    
        assets[ticker] = data
    return assets


def fix_date_columns(assets, date_col='Date', date_format='%Y-%m-%d'):    
    for key in assets.keys(): 
        print(f'Fixing data for {key}')
        assets[key].reset_index(inplace=True)
        assets[key][date_col] = (
            assets[key][date_col]
            .apply(lambda x: dt.datetime.strftime(x, date_format))
        )

        
def read_yfinance_data(tickers, n_years=1, date_col='Date', date_format='%Y-%m-%d'): 
    assets = gather_data(tickers, n_years)
    fix_date_columns(assets)
    return assets
