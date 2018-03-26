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
import time

class SampleStrategy(PortfolioGenerator):

    def __init__(self):
        #insert model weights in here (Neural Net weights)
        
        #stock indices are loaded here 
        self.stock_indices = np.load("preloaded_data/stock_indices.npy")
        self.buffer_size = 100 #look back period
        #self.ticker_data = self.create_buffer_data()
        # number of simulations
        self.num_sims = 10000
    
    def create_buffer_data(self):
        ticker_data = pd.read_csv('original_stock_data/ticker_data.csv')
        ticker_df = self.shape_data(ticker_data)
        #only want to keep the buffer size of lookback data
        ticker_df = ticker_df[len(ticker_df.index)-1-self.buffer_size:len(ticker_df.index)-1]
        return ticker_df
    
    #to shape data into a nice time-series dataframe
    def shape_data(self,stock_features):
        ticker_df = stock_features[['index','returns','ticker']]
        ticker_df = ticker_df.pivot(columns='ticker', values='returns')
        return ticker_df
        
    def build_signal(self, stock_features):
        ticker_df = (self.shape_data(stock_features)).dropna()
        selected_ticker_df = ticker_df[list(self.stock_indices)]
        cov_mat = np.cov(selected_ticker_df,rowvar=False)
        #random mu (before LSTM implementation)
        
        
        mean_vec = np.ones(len(self.stock_indices))/100
        sim_dataframe = pd.DataFrame(columns=["Weights","Sharpe"])
        for i in range(self.num_sims):
            weights = np.random.uniform(low=-1,high=1,size=len(self.stock_indices))
            #normalize weights
            weights = weights/np.std(weights)
            w_t_dot_cov_mat = np.reshape(np.dot(weights.transpose(),cov_mat),(1,300))
            sigma = math.sqrt(np.dot(w_t_dot_cov_mat,weights))
            #print(sigma)
            mu = np.dot(weights,mean_vec)
            sharpe_sample = mu/sigma
            sim_dataframe.at[i,"Weights"]=weights
            sim_dataframe.at[i,"Sharpe"]=sharpe_sample
        
        max_weights = sim_dataframe.loc[np.array(sim_dataframe["Sharpe"]).argmax()]["Weights"]
        
        #need to piece back max weights into a 1000x1 vector
        all_weights = np.zeros(1000)
        for entry in range(len(self.stock_indices)):
            ref_index = self.stock_indices[entry]
            all_weights[ref_index] = max_weights[entry] 
            
        return all_weights



# Test out performance by running 'python sample_strategy.py'
if __name__ == "__main__":
    start = time.time()
    portfolio = SampleStrategy()
    sharpe = portfolio.simulate_portfolio()
    print("*** Strategy Sharpe is {} ***".format(sharpe))
    print("Total Time Taken",str((time.time() - start )*1000.0))