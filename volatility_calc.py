import math
from scipy.stats import norm
from scipy.optimize import brentq
import numpy as np
import yfinance as yf
from datetime import datetime
import pandas as pd

def get_risk_free_rate():
    try:
        treasury_10y = yf.Ticker("^TNX")
        hist = treasury_10y.history(period="1d")
        if not hist.empty:
            rate = hist['Close'].iloc[-1] / 100.0
            return rate
        
        treasury_3m = yf.Ticker("^IRX")
        hist = treasury_3m.history(period="1d")
        if not hist.empty:
            rate = hist['Close'].iloc[-1] / 100.0
            return rate
        
        treasury_1m = yf.Ticker("^BIL")
        hist = treasury_1m.history(period="1d")
        if not hist.empty:
            price = hist['Close'].iloc[-1]
            rate = (100 - price) / 100 * 12
            return rate / 100.0
        
        return 0.05
        
    except Exception as e:
        print(f"Warning: Could not fetch risk-free rate: {e}")
        return 0.05

def get_market_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        info = ticker.fast_info
        spot_price = info.get('last_price') or info.get('lastPrice')
        if spot_price is None:
            hist = ticker.history(period="1d")
            if not hist.empty:
                spot_price = hist['Close'].iloc[-1]
        
        dividend_yield = 0.0
        try:
            dividend_yield = info.get('dividend_yield', 0.0)
            if dividend_yield is None:
                dividend_yield = 0.0
        except:
            pass
        
        risk_free_rate = get_risk_free_rate()
        
        return {
            'spot_price': spot_price,
            'dividend_yield': dividend_yield,
            'risk_free_rate': risk_free_rate
        }
    except Exception as e:
        print(f"Warning: Could not fetch market data: {e}")
        return {
            'spot_price': None,
            'dividend_yield': 0.0,
            'risk_free_rate': 0.05
        }

def call_price_black_scholes(S, K, T, r, q, vol):
    if T <= 0:
        return max(S - K, 0.0)
    if vol < 1e-12:
        return math.exp(-q*T) * max(S - K * math.exp(-r*T), 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S/K) + (r - q + 0.5 * vol**2) * T) / (vol * sqrtT)
    d2 = d1 - vol * sqrtT
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    present_S = S * math.exp(-q * T)
    present_K = K * math.exp(-r * T)
    return present_S * Nd1 - present_K * Nd2

def implied_volatility(price, S, K, T, r, q=0.0, method='brentq'):
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    
    intrinsic = max(S * math.exp(-q*T) - K * math.exp(-r*T), 0.0)
    
    if price < intrinsic - 1e-6:
        return None
    
    if abs(price - intrinsic) < 1e-6:
        return 1e-6
    
    def price_diff(vol):
        return call_price_black_scholes(S, K, T, r, q, vol) - price
    
    vol_lower = 1e-8
    vol_upper = 5.0
    
    if price_diff(vol_lower) * price_diff(vol_upper) > 0:
        vol_upper = 10.0
        if price_diff(vol_lower) * price_diff(vol_upper) > 0:
            return None
    
    try:
        if method == 'brentq':
            iv = brentq(price_diff, vol_lower, vol_upper, maxiter=100, xtol=1e-8, rtol=1e-8)
        else:
            from scipy.optimize import newton
            iv = newton(price_diff, x0=0.3, maxiter=100, tol=1e-8)
        
        if iv < 0 or iv > 10.0:
            return None
        return iv
    except Exception as e:
        return None

def calculate_implied_volatility_with_market_data(options_df, ticker_symbol, use_american_adjustment=True):
    market_data = get_market_data(ticker_symbol)
    spot_price = market_data['spot_price']
    dividend_yield = market_data['dividend_yield']
    risk_free_rate = market_data['risk_free_rate']
    
    if spot_price is None:
        print("Warning: Could not determine spot price, using fallback")
        spot_price = options_df['strike'].median()
    
    df = options_df.copy()
    
    if 'bid' in df.columns and 'ask' in df.columns:
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        df['call_price'] = df['ask']
        
        df['spread_pct'] = (df['ask'] - df['bid']) / df['mid_price']
        df = df[df['spread_pct'] < 0.5]
    elif 'lastPrice' in df.columns:
        df['call_price'] = df['lastPrice']
    else:
        print("Warning: No pricing data available")
        return df
    
    if use_american_adjustment:
        df['moneyness'] = (spot_price - df['strike']) / spot_price
        df = df[df['moneyness'] < 0.2]
    
    def calc_iv(row):
        return implied_volatility(
            price=row['call_price'],
            S=spot_price,
            K=row['strike'],
            T=row['days_to_expiry'] / 252.0,
            r=risk_free_rate,
            q=dividend_yield
        )
    
    df['imp_vol'] = df.apply(calc_iv, axis=1)
    
    df['spot_price'] = spot_price
    df['dividend_yield'] = dividend_yield
    df['risk_free_rate'] = risk_free_rate
    
    return df

def filter_quality_options(df, min_volume=0, max_spread_pct=0.3):
    filtered_df = df.copy()
    
    if 'volume' in filtered_df.columns and min_volume > 0:
        filtered_df = filtered_df[filtered_df['volume'] >= min_volume]
    
    if 'spread_pct' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['spread_pct'] <= max_spread_pct]
    
    filtered_df = filtered_df[
        (filtered_df['imp_vol'] >= 0.01) &
        (filtered_df['imp_vol'] <= 2.0)
    ]
    
    filtered_df = filtered_df[filtered_df['days_to_expiry'] >= 1]
    
    return filtered_df

def calculate_term_structure_iv(df, spot_price):
    atm_options = []
    
    for expiry in df['days_to_expiry'].unique():
        exp_data = df[df['days_to_expiry'] == expiry]
        if len(exp_data) == 0:
            continue
            
        atm_idx = (exp_data['strike'] - spot_price).abs().idxmin()
        atm_option = exp_data.loc[atm_idx]
        
        atm_options.append({
            'days_to_expiry': expiry,
            'strike': atm_option['strike'],
            'atm_iv': atm_option['imp_vol'],
            'moneyness': abs(atm_option['strike'] - spot_price) / spot_price
        })
    
    term_structure = pd.DataFrame(atm_options)
    term_structure = term_structure.sort_values('days_to_expiry')
    
    return term_structure

def validate_implied_volatility(df):
    issues = []
    
    missing_iv = df['imp_vol'].isna().sum()
    if missing_iv > 0:
        issues.append(f"{missing_iv} options have missing implied volatility")
    
    extreme_high = (df['imp_vol'] > 2.0).sum()
    extreme_low = (df['imp_vol'] < 0.01).sum()
    
    if extreme_high > 0:
        issues.append(f"{extreme_high} options have extremely high IV (>200%)")
    if extreme_low > 0:
        issues.append(f"{extreme_low} options have extremely low IV (<1%)")
    
    if len(df) > 10:
        for expiry in df['days_to_expiry'].unique():
            exp_data = df[df['days_to_expiry'] == expiry]
            if len(exp_data) > 5:
                atm_strike = exp_data['strike'].iloc[(exp_data['strike'] - exp_data['spot_price'].iloc[0]).abs().argsort()[:1]]
                if len(atm_strike) > 0:
                    atm_iv = exp_data[exp_data['strike'] == atm_strike.iloc[0]]['imp_vol'].iloc[0]
                    if atm_iv is not None:
                        far_options = exp_data[abs(exp_data['strike'] - atm_strike.iloc[0]) / atm_strike.iloc[0] > 0.2]
                        if len(far_options) > 0:
                            far_ivs = far_options['imp_vol'].dropna()
                            if len(far_ivs) > 0:
                                if (far_ivs > atm_iv * 3).any():
                                    issues.append(f"Some {expiry}-day options have IV > 3x ATM IV")
    
    return issues
