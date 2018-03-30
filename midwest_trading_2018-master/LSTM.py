
import numpy as np
import matplotlib.pyplot as plt
import math
import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input


MSE = np.load("MSE.npy")
MEAN_RATIO = np.load("MEAN_RATIO.npy")

def train_1000():
	global MSE, MEAN_RATIO
	for stock_index in range(600,1000):

		file_name = "stock_data/stock_" + str(stock_index) + ".npy"

		stock_m = np.load(file_name)


		TRAIN_SIZE = 2000

		NUM_INPUT = 15
		SHIFT = True

		# train_x = stock_m[:TRAIN_SIZE]
		# train_y =  stock_m[1:TRAIN_SIZE + 1]

		# test_x,  test_y = stock_m[TRAIN_SIZE:-1], stock_m[TRAIN_SIZE + 1:]

		X = np.hstack((stock_m[:,0:3], stock_m[:,4:])).astype(np.float64)
		x = np.zeros((X.shape[0], X.shape[1]), dtype = np.float64)
		for i in range(1,X.shape[0]):
			x[i-1,:] = (X[i,:] - X[i-1,:])

		Y = stock_m[:,3].astype(np.float64)
		# for i in range(X.shape[1]):
		# 	mean = np.mean(X[:,i])
		# 	std = np.std(X[:,i])
		# 	if std != 0:
		# 		X[:,i] = (X[:,i] - mean)/std
		# 	else:
		# 		X[:,i] = (X[:,i] - mean)

		# Y = (Y-np.mean(Y))/np.std(Y)
		if SHIFT:
			print("Shifted!!!!!!!!!!!!!!")
			train_x = x[0:TRAIN_SIZE-1,:]
			train_y = Y[1:TRAIN_SIZE]
			# train_y = train_y.reshape(train_y.shape[0],1)

			test_x = x[TRAIN_SIZE-1:-1, :]
			test_y = Y[TRAIN_SIZE:]
		else:
			train_x = x[1:TRAIN_SIZE,:]
			train_y = Y[1:TRAIN_SIZE]
			# train_y = train_y.reshape(train_y.shape[0],1)

			test_x = x[TRAIN_SIZE:, :]
			test_y = Y[TRAIN_SIZE:]



		# test_y = test_y.reshape(test_y.shape[0],1)
		# print(train_x[:10,:])
		# print("=======================================")
		# print(test_x[:10,:])
		# print("================================")
		# print(Y[:10])
		# reshape input to be [samples, time steps, features]

		model = Sequential()
		# model.add(Dense(128,  input_shape=(train_x.shape[1],)))
		# model.add(Dropout(0.5))
		model.add(Dense(1024,  input_shape=(train_x.shape[1],)))
		model.add(Dropout(0.8))
		# model.add(Dense(256))
		# model.add(Dropout(0.8))
		# model.add(Dense(128))
		# model.add(Dropout(0.5))
		# model.add(Dense(64))
		# model.add(Dropout(0.5))
		# model.add(Dense(16))
		# model.add(Dropout(0.5))
		# model.add(Dense(64, input_shape=(train_x.shape[1],)))
		# model.add(Dropout(0.5))
		model.add(Dense(1))
		# callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=3, verbose=0, mode='auto')
		adm = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=5e-5)
		model.compile(loss='mean_squared_error', optimizer=adm)
		model.fit(train_x, train_y, epochs=6000, batch_size=2000, validation_data=(test_x, test_y), verbose = 0, shuffle = False)

		trainPredict = model.predict(train_x)
		testPredict = model.predict(test_x)

		print("train_y max", train_y.argmax())
		print("train predict max", trainPredict.argmax())
		# invert predictionsa = model.
		trainPredict = trainPredict.reshape(1, trainPredict.shape[0])[0,:]
		testPredict = testPredict.reshape(1,testPredict.shape[0])[0,:]
		# print(testPredict.shape)
		# test_y = test_y

		pred_positive_pos = (train_y> 0)
		pred_pos = trainPredict[pred_positive_pos]
		actu_pos = train_y[pred_positive_pos]

		# pred_same_diff = np.sum(pred_pos - actu_pos)
		# pred_diff_diff = np.sum(pred_pos - np.abs(actu_pos))

		pred_pos_sum = np.sum(pred_pos)
		actu_pos_sum = np.sum(actu_pos)

		# same_size = True

		# if pred_same_diff > pred_diff_diff:
		# 	same_size = False

		# # if same side
		# if same_side:
		# # mean_ratio = abs(np.sum(interm) / train_x.shape[0])
		mean_ratio = pred_pos_sum / actu_pos_sum


		trainPredict = trainPredict / mean_ratio
		testPredict = testPredict / mean_ratio

		MEAN_RATIO[stock_index-600] = mean_ratio
		_eval = np.sum(np.square(testPredict - test_y))/train_y.shape[0]
		MSE[stock_index-600] = _eval

		model.save("models/stock_" + str(stock_index) + "_model.h5")
		np.save('MSE.npy', MSE)
		np.save("MEAN_RATIO.npy", MEAN_RATIO)


		print("Finish training stock: ", stock_index)
		print("MSE: ", _eval)

		x = [ i for i in range(testPredict.shape[0])]

		plt.figure(1)
		plt.subplot(211)
		plt.plot(x[:40], testPredict[:40])

		plt.subplot(212)
		plt.plot(x[:40],test_y[:40])

		plt.show()
		x = [ i for i in range(trainPredict.shape[0])]
		plt.figure(1)
		plt.subplot(211)
		plt.plot(x, trainPredict)

		plt.subplot(212)
		plt.plot(x,train_y)

		plt.show()


train_1000()
print("Done training")



		# convert_model = Sequential()
		# convert_model.add(Dense(128,activation = "linear", input_shape = (1,)))
		# model.add(Dropout(0.8))
		# convert_model.add(Dense(1, activation = 'linear'))
		# convert_model.compile(loss='mean_squared_error', optimizer='adam')
		# convert_model.fit(trainPredict, train_y, epochs=1000, batch_size=500, verbose = 2, shuffle = False)

		# a = Input(shape=(1,))
		# b = Dense(16)(a)
		# convert_model = Model(inputs=a, outputs=b)
		# convert_model.compile(loss='mean_squared_error', optimizer='adam')
		# convert_model.fit(trainPredict, train_y, epochs=500, batch_size=1000,  verbose = 2, shuffle = False)

		# convert_model = LinearRegression()
		# convert_model.fit(trainPredict.reshape(trainPredict.shape[0], 1),train_y)

		# testPredict = convert_model.predict(testPredict.reshape(testPredict.shape[0], 1)).reshape(1,testPredict.shape[0])[0,:]
		# trainPredict = convert_model.predict(trainPredict.reshape(trainPredict.shape[0], 1)).reshape(1,trainPredict.shape[0])[0,:]

		# trainPredict = trainPredict / 1000
		# testPredict = testPredict / 1000
		# train_y = train_y / 1000
		# test_y = test_y / 1000


		# print(testPredict[:20])
		# print(np.square(testPredict - test_y).shape)
		# _eval = np.sum(np.square(testPredict - test_y))/train_y.shape[0]
# print(_eval)

# print("-------------- test_y ---------------")
# print(test_y[:20])
# print("------------------ testPredict --------------------")
# print(testPredict[:20])
# print("--- train converter---")



# x = [ i for i in range(testPredict.shape[0])]

# plt.figure(1)
# plt.subplot(211)
# plt.plot(x, testPredict)

# plt.subplot(212)
# plt.plot(x,test_y)

# plt.show()
# x = [ i for i in range(trainPredict.shape[0])]
# plt.figure(1)
# plt.subplot(211)
# plt.plot(x, trainPredict)

# plt.subplot(212)
# plt.plot(x,train_y)

# plt.show()