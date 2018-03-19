import numpy as np 
import math

def calculate_cumulative_return(arr):
    """
    sum up all the return value together
    - arr: a 1d array with retur values and 
    """
    ret = np.sum(arr)
    print("Cumulative return of the stock is: ", ret, "for ", arr.shape[0], "timesteps")
    return ret

def calculate_cumulative_positive_return(arr):
    """
    only sum up all the positive value of the array
    - arr: a 1d array of return values
    """
    pos_ret_arr = arr[arr>=0]
    pos_ret = np.sum(pos_ret_arr)

    print("Cumulative positive return of the stock is: ", pos_ret, "for ", arr.shape[0], "timesteps")
    return pos_ret


def calculate_estimate_mse(actu_arr, pred_arr):
    """
    calculate the MSE of the return predicted

    - actu_arr: 1d array with true return value
    - pred_arr: 1d array with predicted values
    """
    diff_arr = actu_arr - pred_arr
    mse = np.sum(np.square(diff_arr))/actu_arr.shape[0]
    print("Estimate MSE is: ", mse, "for", actu_arr.shape[0], "timesteps")
    return mse

def calculate_estimate_error_std(actu_arr, pred_arr):

    """
    calculate the std of the estimation error

    """
    diff_arr = actu_arr - pred_arr
    mean = np.sum(diff_arr) / actu_arr.shape[0]
    diff_arr = diff_arr - mean
    std = math.sqrt(np.sum(np.square(diff_arr)) / (actu_arr.shape[0] - 1))
    print("The standard deviation of prediction error is: ", std, "for", actu_arr.shape[0], "timesteps")
    return std
    

def calculate_loss(actu_arr, pred_arr):
    """
    calculate the loss of the prediction due to predict positive return but result in negative return
    The value is cumulative of all loss
    single entry loss = sum (actu_return)  where actu < 0 and pred > 0
    """

    actu_neg_mask = actu_arr < 0
    pred_pos_mask = pred_arr > 0
    # calculate mask where actu = 1 and pred = 1
    new_mask = np.logical_and(actu_neg_mask, pred_pos_mask)
    actu_ret_from_mask = actu_arr[new_mask]
    cum_loss = np.sum(actu_ret_from_mask)
    print("The cumulative loss result in prediction error is ", cum_loss, "for", actu_arr.shape[0], "timesteps")
