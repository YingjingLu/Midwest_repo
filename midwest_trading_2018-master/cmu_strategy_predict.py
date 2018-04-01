import sklearn
import pickle
import keras
import pprint
import math
import numpy as np 
from sklearn.externals import joblib

def load_model(KERAS = False, RIDGE = True):

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


# class fake_model(object):
#     def __init__(self):
#         self.x = 0.001
#     def predict(self, data):
#         return self.x

# def load_model():
#     model_list = []
#     for i in range(1000):
#         model_list.append(fake_model())
#     return model_list

def select_stocks_from_industry(percentage):
    assert(0 < percentage < 1)
    selected_stock = []
    for industry in range(1,6):
        industry_rank = np.load("ind_rank/rankforind_" + str(industry) + ".npy")

        num = math.floor(industry_rank.shape[0]*percentage)
        selected = list(industry_rank[:num])
        selected_stock += selected
    return np.array(selected_stock)


def model_predict(processed_data, model, KERAS = False):
    if KERAS:
        return model.predict(processed_data)[0]
    else:
        # perform ridge regression and scale up
        return model.predict(processed_data)[0]*10

def preprocess_data_one_stock(stock_features, stock_idx, LSTM = False):
    if type(stock_idx) != type(1):
        stock_idx = int(stock_idx)
    if stock_idx >= 1000 or stock_idx < 0:
        print("Error invalid stock index, set to 0")
        stock_idx = 0

    total_matrix = np.array(stock_features.values)
     # strip out the index column in original data
    stock_matrix = (total_matrix[total_matrix[:,5] == stock_idx])[:,1:]
    stock_matrix = np.hstack((stock_matrix[:,0:3], stock_matrix[:,4:])).astype(np.float64)
    # print("Stock matrix shape", stock_matrix.shape)
    data_lookback, _ = stock_matrix.shape

    # delta data of the last frame
    data = (stock_matrix[-1, :] - stock_matrix[-2,:]).reshape(1, stock_matrix.shape[1])
    if LSTM:
        data = np.reshape(data, (data.shape[0],1, data.shape[1]))
    return data


def predict_from_stock_index_list(stock_features, stock_index_list, model_list):
    return_list = []
    for stock_index in stock_index_list:
        data =  preprocess_data_one_stock(stock_features, stock_index)
        model = model_list[stock_index]
        miu = model_predict(data, model)
        return_list.append(miu)
    return return_list
