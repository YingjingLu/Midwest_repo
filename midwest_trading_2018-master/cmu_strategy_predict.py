import sklearn
import pickle
import keras
import math
import numpy as np 
from sklearn.externals import joblib

INDUSTRY_NAME_TO_INDEX = {
    'TECH': 1.0,
    'AGRICULTURE': 2.0,
    'FINANCE': 3.0,
    'CONSUMER': 4.0,
    'OTHER': 5.0
}

def load_model(KERAS = False, RIDGE = True):
    """
    Inputs:
     - KERAS (bool): whether we use keras as the model for prediction
     - RIDGE (bool): whether we use RIDGE regression as the model
    KERAS has the piority if specified True

    Output:
     - model_list (list of length 1000): contains models for each stock corresponding to index

    Assume:
     - models are in models/ folder and the naming contention is as below
    """
    model_list = []
    if KERAS:
        for stock_idx in range(1000):
            model_file_name = "models/stock_"+str(stock_idx)+"_model.h5"
            model = keras.models.load_model(model_file_name)
            model_list.append(model)
    elif RIDGE:
        for stock_index in range(1000):
            model_file_name = "models/model_"+str(stock_index)+".pkl"
            model = joblib.load(model_file_name)
            model_list.append(model)
    return model_list

def select_stocks_from_industry(percentage):

    """
    Input:
     - percentage (float): this should be a float between 0 to 1 exclusive that
                        specifies how much percentage of stock we choose from each industry

    output:
     - a concatenated np array of stock indicies we choose based on percentage

    Assume:
     - rank are stored in ind_rank/ folder and naming convention is as below
    """
    assert(0 < percentage < 1)
    selected_stock = []
    for industry in range(1,6):
        industry_rank = np.load("ind_rank/rankforind_" + str(industry) + ".npy")
        # get the cut off of number of stocks
        num = math.floor(industry_rank.shape[0]*percentage)
        # concatenate stock list
        selected = list(industry_rank[:num])
        selected_stock += selected
    return np.array(selected_stock)


def model_predict(processed_data, model, KERAS = False):

    """
    Input:
     - processed_data (original pandas data frame): a pandas dataframe file as specified in the sample strategy 
                          contains all stocks for certain look back period
     - model(keras or pickle model): a model for a chosen stock from load_models
     - KERAS (bool): whether we use KERAS, if False, we use ridge regression which predicts and scale the prediction by 10

    output:
     - a float number of the predicted miu in the future period
    """

    if KERAS:
        return model.predict(processed_data)[0]
    else:
        # perform ridge regression and scale up
        return model.predict(processed_data)[0]*10

def preprocess_data_one_stock(stock_features, stock_idx, LSTM = False):
    """
    Input:
     - stock_features(original pandas dataframe)
     - stock_idx(int): stock index between 0 to 999 inclusive
     - LSTM(bool): whether we use LSTM format data
    Output:
     - data(ndarray (1,15)): the nd array of delta value of stock factor and global factors

    Assumtion:
     - stock industry index: 1
     - timestamp index : 0
     - at least two timesteps of lookback
    """
    global INDUSTRY_NAME_TO_INDEX
    if type(stock_idx) != type(1):
        stock_idx = int(stock_idx)
    if stock_idx >= 1000 or stock_idx < 0:
        print("Error invalid stock index, set to 0")
        stock_idx = 0
    # the last index of the dataframe we want to keep
    LAST_INDEX = -2
    total_matrix = np.array(stock_features.values)
     # strip out the index column in original data
    stock_matrix = (total_matrix[total_matrix[:,5] == stock_idx])[LAST_INDEX:,1:]
    # convert industry name to number
    for i in range(stock_matrix.shape[0]):
        industry_name = stock_matrix[i,0]
        stock_matrix[i,0] = INDUSTRY_NAME_TO_INDEX[industry_name]
    # strip out the return 
    stock_matrix = np.hstack((stock_matrix[:,0:3], stock_matrix[:,4:])).astype(np.float64)

    # delta data of the last frame
    data = (stock_matrix[-1, :] - stock_matrix[-2,:]).reshape(1, stock_matrix.shape[1])
    if LSTM:
        data = np.reshape(data, (data.shape[0],1, data.shape[1]))
    return data


def predict_from_stock_index_list(stock_features, stock_index_list, model_list):
    """
    Input:
     - stock_features (original pandas dataframe)
     - stock_index_list (list): list of the chosen stock
     - model_list (list of len 1000): the list of models index corresponding to ticker id
    Output:
     - the list of predicted miu which order is the same as the stock_index_list
    """

    return_list = []
    for stock_index in stock_index_list:
        data =  preprocess_data_one_stock(stock_features, stock_index)
        model = model_list[stock_index]
        miu = model_predict(data, model)
        return_list.append(miu)
    return return_list
