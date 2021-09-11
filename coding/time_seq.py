import copy
import os
import threading

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm

import matplotlib.pyplot as plt

from coding.state_model import draw_acf_pacf, testStationarity

os.chdir(r'D:\2021CUMCM\coding')

df_loss = pd.read_excel('./../quiz/C/附件2 近5年8家转运商的相关数据.xlsx')
df_loss_data = df_loss.iloc[:,1:]

one_transfer = df_loss_data.iloc[0,:]
copy_t = copy.deepcopy(one_transfer)
copy_t_diff_1 = copy_t.diff(100)
copy_t_diff_1.dropna(inplace=True)
# log_t = np.log(copy_t)
# log_t_diff_1 = log_t.diff(1)
# log_t_diff_1.dropna(inplace=True)
# draw_acf_pacf(log_t_diff_1)
#
# testStationarity(copy_t_diff_1)
# model = ARMA(copy_t_diff_1, order=(1, 1))
# result_arma = model.fit(disp=-1, method='css')
# predict_ts  = result_arma.predict()
# shift_diff_1 = copy_t.shift(1)
# shift_diff_1.dropna(inplace=True)
# recover = predict_ts.add(shift_diff_1)

# ts = copy_t[recover.index]
# ts.plot()
# recover.plot()

fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig=sm.graphics.tsa.plot_acf(copy_t_diff_1,lags=40,ax=ax1)
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(copy_t_diff_1,lags=40,ax=ax2)
plt.show()

arma_mod01=sm.tsa.ARMA(copy_t_diff_1,(1,1)).fit()
resid = arma_mod01.resid

fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig=sm.graphics.tsa.plot_acf(resid.values.squeeze(),lags=40,ax=ax1) #squeeze()数组变为1维
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(resid,lags=40,ax=ax2)
plt.show()


threading.Thread()