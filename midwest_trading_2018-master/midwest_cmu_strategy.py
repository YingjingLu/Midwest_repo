#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 22:33:59 2018

@author: wengshian
"""

from portfolio import PortfolioGenerator
import pandas as pd
import numpy as np

class SampleStrategy(PortfolioGenerator):

    def __init__(self):
        #insert model weights in here (Neural Net weights)
        
        #stock indices are loaded here 
        self.stock_indicies = np.load("preloaded_data/stock_indices.npy")
        self.buffer_size = 100 #look back period
        self.ticker_data = create_buffer_data()
    
    def create_buffer_data(self):
        ticker_df = pd.read_csv('stock_data/ticker_data.csv')
        ticker_df = ticker_df[['index','returns','ticker','timestep']]
        ticker_df = ticker_df.pivot(index='timestep', columns='ticker', values='returns')
        #only want to keep the buffer size of lookback data
        ticker_df = ticker_df[len(ticker_df.index)-1-self.buffer_size:len(ticker_df.index)-1]
        return ticker_df
        

    def build_signal(self, stock_features):
        return self.momentum(stock_features)

    def momentum(self, stock_features):
        return stock_features.groupby(['ticker'])['returns'].mean()

# Test out performance by running 'python sample_strategy.py'
if __name__ == "__main__":
    portfolio = SampleStrategy()
    #sharpe = portfolio.simulate_portfolio()
    #print("*** Strategy Sharpe is {} ***".format(sharpe))