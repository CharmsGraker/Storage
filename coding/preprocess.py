import numpy as np
import pandas as pd


def getLossMatrix(inverse=True):
    """

    :return:
    """
    # pd.DataFrame(np.zeros((future_step, len(TRANSFOR))))
    sheet2_name = './../quiz/C/附件2 近5年8家转运商的相关数据.xlsx'

    df_loss = pd.read_excel(sheet2_name)
    df_loss.drop(columns=['转运商ID'], inplace=True)
    if inverse:
        df_loss = df_loss.iloc[:, :24]
        df_loss = pd.DataFrame(np.mat(df_loss.values).T)
    loss_matrix = df_loss
    return loss_matrix / 100


def getSatisfyMatrix():
    sheet1_name = './../quiz/C/附件1 近5年402家供应商的相关数据.xlsx'

    df_order = pd.read_excel(sheet1_name, sheet_name=0)
    df_offer = pd.read_excel(sheet1_name, sheet_name=1)
    eps = 0.001
    sat = (df_order.iloc[:, 2:] + eps) / (df_offer.iloc[:, 2:] + eps)
    s_matrix = pd.DataFrame(np.mat(sat.values[:, :24]).T)
    return s_matrix
    pass
