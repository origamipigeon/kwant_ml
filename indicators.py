import pandas as pd
import numpy as np 			  		 			 	 	 		 		 	  		   	  			  	
import datetime as dt


def compute_indicators(df_prices, start_date, end_date, lookback=20):
    """
    util function to calc the indicators for a given set of stocks
    @df_prices : NxM list of stocks with closing prices
    @start_time : when to start indicators
    @end_date : when to end indicators
    @lookback : number of days to look back for indicators such as bolinger bands
    """
    # df_prices = df_all_prices[symbol]
    # df_prices = df_prices.fillna(method='ffill')
    # df_prices = df_prices.fillna(method='bfill')
    df_normalised_prices = df_prices / df_prices.iloc[0]
    
    df_indicator_values = pd.DataFrame(index=df_prices.index)
    
    # indicator 1, Simple moving average, 20 days
    df_rolling = df_normalised_prices.rolling(window=lookback,center=False)

    df_sma = df_rolling.mean().dropna()
    st = df_sma.index.searchsorted(start_date)
    ed = df_sma.index.searchsorted(end_date)
    df_sma = df_sma[st:ed]
    
    df_indicator_values['sma'] = df_sma
    
    # indicator 2 momentum
    df_momentum = (df_normalised_prices.diff(lookback)/df_normalised_prices.shift(lookback)) 
    df_momentum = df_momentum.dropna()
    st = df_momentum.index.searchsorted(start_date)
    ed = df_momentum.index.searchsorted(end_date)
    df_momentum = df_momentum[st:ed]
    df_indicator_values['momentum'] = df_momentum
    
    # indicator 3 volatility
    df_rolling_std = df_rolling.std(window=lookback).dropna()
    st = df_rolling_std.index.searchsorted(start_date)
    ed = df_rolling_std.index.searchsorted(end_date)
    df_rolling_std = df_rolling_std[st:ed]
    # save std as volitility
    df_indicator_values['volatility'] = df_rolling_std
    
    # indicator 4 bollinger bands + position
    df_upper = df_sma + (2 * df_rolling_std)
    df_lower = df_sma - (2 * df_rolling_std)
    df_indicator_values['bb_upper'] = df_upper
    df_indicator_values['bb_lower'] = df_lower
    df_indicator_values['bbp'] = ((df_normalised_prices - df_lower) / (df_upper - df_lower))
    
    # slice to requested date range
    df_indicator_values = df_indicator_values.dropna()
    
    return df_indicator_values

if __name__ == "__main__":
    None
