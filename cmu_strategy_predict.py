import sklearn
import keras
import pprint
import numpy as np 
def load_model():

    model_list = []

    for stock_idx in range(1000):
        model_file_name = "models/stock_"+str(stock_idx)+"_model.h5"
        model = keras.models.load_model(model_file_name)
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

def model_predict(processed_data, model):
    return model.predict(processed_data)

def preprocess_data_one_stock(stock_features, stock_idx, look_back, LSTM = False):
    if type(stock_idx) != type(1):
        stock_idx = int(stock_idx)
    if stock_idx >= 1000 or stock_idx < 0:
        print("Error invalid stock index, set to 0")
        stock_idx = 0

    total_matrix = np.array(stock_features.values)
     # strip out the index column in original data
    stock_matrix = (total_matrix[total_matrix[:,5] == stock_idx])[:,1:]

    print("Stock matrix shape", stock_matrix.shape)
    data_lookback, _ = stock_matrix.shape

    if not ( 1<= look_back <= data_lookback):
        print("Error: invalid lookback, set to 1")
        look_back = 1
    data = stock_matrix[-1*look_back :, :]
    if LSTM:
        data = np.reshape(data, (data.shape[0],1, data.shape[1]))
    return data


def predict_from_stock_index_list(stock_features, stock_index_list, model_list, look_back):
    return_list = []
    for stock_index in stock_index_list:
        data =  preprocess_data_one_stock(stock_features, stock_index, look_back)
        model = model_list[stock_index]
        miu = model_predict(data, model)
        return_list.append(miu)
    return return_list
