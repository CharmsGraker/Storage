import numpy as np
import pandas as pd
import os
from xgboost import XGBRegressor

from coding.utils import splitSlideWindow, splitValidation

xg_reg = XGBRegressor()



os.chdir(r'D:\2021CUMCM\coding')
sheet1_name = './../quiz/C/附件1 近5年402家供应商的相关数据.xlsx'

df_order = pd.read_excel(sheet1_name, sheet_name=0)
df_offer = pd.read_excel(sheet1_name, sheet_name=1)



step = 1
features = [0]
n_features = len(features)
window_size = 60
X_data = np.array(df_order.iloc[:,2:].values).T
Xdataset,ydataset,y_scaler = splitSlideWindow(X_data,window_size=window_size,features=features,step=step)
X_train,X_val,y_train,y_val = splitValidation(Xdataset,ydataset,validate_split=140)



# dataset = df_order.iloc[:, 2:].values
# dataset = np.array(dataset).T
# X = dataset[:-1,0].reshape(-1, 1)
# Y = dataset[1:,0]
xg_reg.fit(X_train,y_train)