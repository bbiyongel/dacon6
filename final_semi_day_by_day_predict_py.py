import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
 
look_back = 1
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


def predcit_next(nptf, column):

    # normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    nptf = nptf[column].values
    nptf = nptf.reshape(-1,1)
    nptf = scaler.fit_transform(nptf)

    train_size = int(len(nptf) * 0.9)
    test_size = len(nptf) - train_size
    train, test = nptf[0:train_size], nptf[train_size:len(nptf)]

    # create dataset for learning
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    # simple lstm network learning
    model = Sequential()
    model.add(LSTM(8, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)

    
    # 향후 10달의 평균 예측
    look_ahead = 10
    res = nptf.copy()
    res = create_dataset(res,1)
    res = res[0]

    res = np.reshape(res, (res.shape[0], 1, res.shape[1]))
    pre = model.predict(res)
    res = np.append(res, [[pre[-1]]])
    res = np.reshape(res, (res.shape[0], 1))

    for _ in range(look_ahead):
        res = np.reshape(res, (res.shape[0], 1, res.shape[1]))
        pre = model.predict(res)
        res = np.append(res, [[pre[-1]]])
        res = np.reshape(res, (res.shape[0], 1))
    pre = scaler.inverse_transform(pre)

    return pre[-1]


# file loader
fullpath = './data/Regular_Season_Batter_Month.csv'
pandf = pd.read_csv(fullpath, index_col='idx')
main_col = [
        'am_avg','avg','AB','R','H',
        '1B','2B','3B','HR','RBI','BB',
        'HBP','SO','GDP','BABIP','SLG',
        'OBP','am_BABIP','am_SLG','am_OBP'
         ]

final_predict = []
id_arr = pandf['batter_id'].unique()
id_arr = id_arr.tolist()
for bat_id in id_arr:
    eachPlayer = [bat_id]
    nptf = pandf.loc[pandf['batter_id'] == bat_id]
    eachPlayer.append(str(nptf['batter_name']))
    for col in main_col: 
        if len(nptf) > 20:
            res = predcit_next(nptf, col)
            eachPlayer.append(res)
        else:
            res = nptf[col].mean()
            eachPlayer.append(res)
    final_predict.append(eachPlayer)
print(final_predict)
