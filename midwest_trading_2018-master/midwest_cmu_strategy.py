#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 22:33:59 2018

@author: wengshian
"""

from portfolio import PortfolioGenerator
from cmu_strategy_predict import *
import pandas as pd
import numpy as np
from scipy.stats import norm
import math 
import time
import keras


class SampleStrategy(PortfolioGenerator):

    def __init__(self):
        #insert model weights in here (Neural Net weights)
        
        #stock indices are loaded here 
        self.stock_indices = select_stocks_from_industry(0.1)
        self.buffer_size = 100 #look back period
        #self.ticker_data = self.create_buffer_data()
        # number of simulations for markowitz
        self.num_sims = 10000
        self.to_use_MCA = True #implement MCA
        self.model_list = load_model()
    
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
    
    def MCA1(self, stock_data,miu):
        global p
        # p - Pearson Matrix (correlation matrix)
        p = np.corrcoef(np.transpose(stock_data))
        assert(stock_data.shape[1]==miu.shape[0])
        # sigma - asset std
        sigma = stock_data.var(axis=0) 

        n = p.shape[0]
        global sign
        sign = np.diag(np.array([1 if i >= 0 else -1 for i in miu]))

        # 1D array of upper triangular elements
        upper_tri = p[np.triu_indices(n,1)]

        num_elem = upper_tri.shape[0]
        miu_p = np.mean(upper_tri)
        sigma_p = np.std(upper_tri)

        # Adjusted p
        p_A = 1 - norm.cdf(p, miu_p, sigma_p)

        # weight after transformation, avg of each row
        w_T = p_A.mean(axis=1)
        w_T = abs(miu)/w_T
        # rank in descending order
        rank = w_T[::-1].argsort().argsort()
        # rank weight
        w_Rank = (rank+1)/sum(rank+1)
        temp = np.dot(w_Rank, p_A)
        w_Rank = temp/sum(temp)
        # scale w_rank by asset std and normalize
        scaled_w = w_Rank/sigma
        global w
        w = scaled_w/sum(scaled_w)
        w = np.dot(w, np.transpose(sign))
        return w

    def MCA2(self, stock_data,miu):
        global p
        # p - Pearson Matrix (correlation matrix)
        p = np.corrcoef(np.transpose(stock_data))
        assert(stock_data.shape[1]==miu.shape[0])
        # sigma - asset std
        sigma = stock_data.var(axis=0) 
        
        n = p.shape[0]
        sign = np.diag(np.array([1 if i >= 0 else -1 for i in miu]))

        w0 = np.mean(p, axis=1)

        miu_0 = np.mean(w0)
        sigma_0 = np.std(w0)

        w_T = 1 - norm.cdf(w0, miu_0, sigma_0)

        w_T = w_T * miu

        rank = w_T[::-1].argsort().argsort()
        w_Rank = (rank+1)/sum(rank+1)
        p_A = 1 - p
        temp = np.dot(w_Rank, p_A)
        w_Rank = temp/sum(temp)
        # scale w_rank by asset std and normalize
        scaled_w = w_Rank/sigma
        global w
        w = scaled_w/sum(scaled_w)
        w = np.dot(w, np.transpose(sign))
        return w
    
        
    def build_signal(self, stock_features):
        global selected_ticker_df
        global input_data
        input_data = stock_features
        ticker_df = (self.shape_data(stock_features)).dropna()
        selected_ticker_df = ticker_df[list(self.stock_indices)]
        cov_mat = np.cov(selected_ticker_df,rowvar=False)
        #random mu (before LSTM implementation)
        if self.to_use_MCA==True:
            selected_ticker_df = np.array(selected_ticker_df)
            miu = predict_from_stock_index_list(stock_features, 
                                                self.stock_indices,
                                                self.model_list)
            miu = np.array(miu)
            # miu = Yingjing's_algo(selected_ticker_df[-1])
            max_weights = self.MCA1(selected_ticker_df,miu)
        
        else:
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
