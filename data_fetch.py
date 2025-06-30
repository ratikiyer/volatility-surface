import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np

def get_options_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    try:
        expirations = ticker.options
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve options for {ticker_symbol}: {e}")
    if not expirations:
        raise RuntimeError(f"No options data found for ticker {ticker_symbol}")
    
    all_calls = []
    today = datetime.today().date()
    
    for exp_date_str in expirations:
        try:
            opt_chain = ticker.option_chain(exp_date_str)
        except Exception as e:
            print(f"Warning: could not fetch data for expiration {exp_date_str}: {e}")
            continue
        calls_df = opt_chain.calls.copy()
        if calls_df is None or calls_df.empty:
            continue
        calls_df['expiration'] = pd.to_datetime(exp_date_str).date()
        calls_df['days_to_expiry'] = (calls_df['expiration'] - today).apply(lambda x: x.days)
        calls_df = calls_df[calls_df['days_to_expiry'] > 0]
        calls_df = calls_df.dropna(subset=['bid', 'ask'])
        calls_df = calls_df[(calls_df['bid'] > 0) & (calls_df['ask'] > 0)]
        all_calls.append(calls_df)
    
    if not all_calls:
        raise RuntimeError(f"Unable to fetch any options data for {ticker_symbol}")
    
    options_data = pd.concat(all_calls, ignore_index=True)
    options_data['strike'] = options_data['strike'].astype(float)
    options_data['days_to_expiry'] = options_data['days_to_expiry'].astype(int)
    options_data['bid'] = options_data['bid'].astype(float)
    options_data['ask'] = options_data['ask'].astype(float)
    options_data.sort_values(['days_to_expiry', 'strike'], inplace=True)
    spot_price = None
    try:
        info = ticker.fast_info
        spot_price = info.get('last_price') or info.get('lastPrice')
        if spot_price is None:
            hist = ticker.history(period="1d")
            if not hist.empty:
                spot_price = hist['Close'].iloc[-1]
    except Exception as e:
        print(f"Warning: could not retrieve spot price for {ticker_symbol}: {e}")
    if spot_price is None:
        try:
            atm_strike = options_data.loc[options_data['days_to_expiry'] == options_data['days_to_expiry'].min(), 'strike'].median()
            spot_price = atm_strike
        except:
            spot_price = 100.0
    
    return options_data, float(spot_price)
