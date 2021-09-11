import numpy as np


def splitSlideWindow(X_data, features, all_feature=8, window_size=50, step=30, norm=False):
    Xdataset = np.zeros(())
    ydataset = np.zeros(())
    n_features = len(features)
    # ctable = copy.deepcopy(table)
    # ctable = ctable.drop(columns=['材料分类'])

    # ctable['Deaths'] = d_scaler.fit_transform(np.array(ctable['Deaths']).reshape(-1,1))
    # df_reverse = ctable.iloc[:,2:]
    if norm:
        from sklearn.preprocessing import MinMaxScaler
        y_scaler = MinMaxScaler()
        X_data = y_scaler.fit_transform(X_data.reshape(-1, 1)).reshape(240, all_feature)
    else:
        y_scaler = []
    origin_data = X_data
    total_samples = X_data.shape[0]
    print(total_samples)  # 时间采样
    for i in range(total_samples):
        if i + step + window_size > total_samples:
            break
        if i == 0:
            Xdataset = origin_data[i:i + window_size, features].reshape(-1, window_size, n_features)
            print(Xdataset.shape)

            # Xdataset = Xdataset[None,:,:]

            ydataset = origin_data[i + window_size:i + window_size + step, features].reshape(-1, step, n_features)
            print("i=0,", Xdataset.shape, ydataset.shape)
            continue

        cur_window_x = origin_data[i:i + window_size, features].reshape(-1, window_size, n_features)
        cur_window_y = origin_data[i + window_size:i + window_size + step, features].reshape(step, n_features)

        # cur_window_x = cur_window_x[None,:,:]
        cur_window_y = cur_window_y[None, :, :]
        # print("curX",cur_window_x.shape)
        # print(window.shape)
        Xdataset = np.concatenate((Xdataset, cur_window_x), axis=0)
        # print(Xdataset.shape)

        ydataset = np.concatenate((ydataset, cur_window_y), axis=0)
        # print(ydataset.shape)
        i += 1
    # y_scaler = d_scaler
    print("滑动窗口后，现在样本集X形状:", Xdataset.shape)
    print("滑动窗口后，现在样本集y形状:", ydataset.shape)

    return Xdataset, ydataset, y_scaler


def splitValidation(X_data, Y_data, validate_split):
    valid = validate_split
    X_train = X_data[:valid, :, :]
    X_val = X_data[valid:, :, :]

    y_train = Y_data[:valid, :]
    y_val = Y_data[valid:, :]
    print('验证集形状', X_val.shape)
    return X_train, X_val, y_train, y_val
