
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression

stock_index = int(input("Stock id: "))

file_name = "stock_data/stock_" + str(stock_index) + ".npy"

stock_m = np.load(file_name)


TRAIN_SIZE = 2000

NUM_INPUT = 15
SHIFT = True
SHIFTED = 1
USE_DELTA = False
# train_x = stock_m[:TRAIN_SIZE]
# train_y =  stock_m[1:TRAIN_SIZE + 1]

# test_x,  test_y = stock_m[TRAIN_SIZE:-1], stock_m[TRAIN_SIZE + 1:]

# X = np.hstack((stock_m[:,0:3], stock_m[:,4:])).astype(np.float64)
X = stock_m[:,0:3].astype(np.float64)
x = np.zeros((X.shape[0], X.shape[1]), dtype = np.float64)

# use delta x
if USE_DELTA:
	for i in range(1,X.shape[0]):
		x[i,:] = (X[i,:] - X[i-1,:])
else:
	x = X

Y = stock_m[:,3].astype(np.float64)

# =========== Data Normalization ===========#
# for i in range(X.shape[1]):
# 	mean = np.mean(X[:,i])
# 	std = np.std(X[:,i])
# 	if std != 0:
# 		X[:,i] = (X[:,i] - mean)/std
# 	else:
# 		X[:,i] = (X[:,i] - mean)

# Y = (Y-np.mean(Y))/np.std(Y)

# ========End Data normalization ==============#

# ========== Shift y ahead of x by 1 ++++++++++
if SHIFT:
	print("Shifted!!!!!!!!!!!!!!")
	train_x = x[1:TRAIN_SIZE-1,:]
	train_y = Y[2:TRAIN_SIZE]
	# train_y = train_y.reshape(train_y.shape[0],1)

	test_x = x[TRAIN_SIZE-1:-1, :]
	test_y = Y[TRAIN_SIZE:]

# model = LinearRegression()
# model.fit(train_x, train_y)

# trainPredict = model.predict(train_x)
# testPredict = model.predict(test_x)

# # invert predictionsa = model.
# trainPredict = trainPredict.reshape(1, trainPredict.shape[0])[0,:]
# testPredict = testPredict.reshape(1,testPredict.shape[0])[0,:]
# _eval = np.sum(np.square(testPredict - test_y))/test_y.shape[0]

# print("MSE", _eval)
# exit(0)

MSE_list = []
ALPHA_list = []
for i in range(10, 500):
	model = Lasso(alpha = 0.1*i)
	model.fit(train_x, train_y)

	trainPredict = model.predict(train_x)
	testPredict = model.predict(test_x)

	# invert predictionsa = model.
	trainPredict = trainPredict.reshape(1, trainPredict.shape[0])[0,:]
	testPredict = testPredict.reshape(1,testPredict.shape[0])[0,:]
	_eval = np.sum(np.square(testPredict - test_y))/test_y.shape[0]

	MSE_list.append(_eval)
	ALPHA_list.append(0.1*i)

plt.figure(1)
plt.plot(ALPHA_list, MSE_list)
plt.show()
print(MSE_list)

print("Finish training stock: ", stock_index)
print("MSE: ", _eval)

x = [ i for i in range(testPredict.shape[0])]

plt.figure(1)

plt.subplot(211)
plt.title("First 60 pred from test")
plt.plot(x[:60], testPredict[:60], 'bs',x[:60], testPredict[:60])

plt.subplot(212)
plt.title("First 60 True from test")
plt.plot(x[:60],test_y[:60], 'bs', x[:60],test_y[:60])

plt.show()
x = [ i for i in range(trainPredict.shape[0])]
plt.figure(1)
plt.subplot(211)
plt.plot(x, trainPredict)

plt.subplot(212)
plt.plot(x,train_y)

plt.show()