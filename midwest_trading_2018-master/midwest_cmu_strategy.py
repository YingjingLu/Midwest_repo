#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 22:33:59 2018

@author: wengshian
"""

from portfolio import PortfolioGenerator
import pandas as pd
import numpy as np
import math 
class SampleStrategy(PortfolioGenerator):

    def __init__(self):
        #insert model weights in here (Neural Net weights)
        
        #stock indices are loaded here 
        self.stock_indices = np.load("preloaded_data/stock_indices.npy")
        self.buffer_size = 100 #look back period
        self.ticker_data = self.create_buffer_data()
        # number of simulations
        self.num_sims = 1000
    
    def create_buffer_data(self,stock_features):
        ticker_data = pd.read_csv('original_stock_data/ticker_data.csv')
        ticker_df = self.shape_data(ticker_data)
        #only want to keep the buffer size of lookback data
        ticker_df = ticker_df[len(ticker_df.index)-1-self.buffer_size:len(ticker_df.index)-1]
        return ticker_df
    
    #to shape data into a nice time-series dataframe
    def shape_data(self,stock_features):
        ticker_df = stock_features[['index','returns','ticker','timestep']]
        ticker_df = ticker_df.pivot(index='timestep', columns='ticker', values='returns')
        return ticker_df
        
    def build_signal(self, stock_features):
        np.set
        ticker_df = shape_data(stock_features)
        selected_ticker_df = ticker_df[list(self.stock_indices)]
        cov_mat = np.cov(selected_ticker_df,rowvar=False)
        #random mu (before LSTM implementation)
        mean_vec = np.ones(len(stock_indices))/100
        sim_dataframe = pd.DataFrame(columns=["Weights","Sharpe"])
        for i in range(self.num_sims):
            weights = np.random.uniform(low=-1,high=1,size=len(portfolio.stock_indicies))
            #normalize weights
            weights = weights/np.std(weights)
            sigma = math.sqrt(np.dot(np.dot(weights,cov_mat),weights.transpose()))
            mu = np.dot(weights,mean_vec)
            sharpe_sample = mu/sigma
            sim_dataframe[i,"Weights"]=weights
            sim_dataframe[i,"Sharpe"]=sharpe_sample
            
            
            
        
        return self.momentum(stock_features)



# Test out performance by running 'python sample_strategy.py'
if __name__ == "__main__":
    portfolio = SampleStrategy()
    #sharpe = portfolio.simulate_portfolio()
    #print("*** Strategy Sharpe is {} ***".format(sharpe))