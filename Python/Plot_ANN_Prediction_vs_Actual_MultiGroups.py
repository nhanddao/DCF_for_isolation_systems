# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 15:32:09 2026

@author: nhand
"""

import numpy as np
from tensorflow import keras
from pickle import load
import os

def load_keras_ensemble(base_path,n_splits=10):
    models = []
    for i in range(1, n_splits+1):
        path = f"{base_path}/ANNModel_fold{i}.keras"
        if os.path.exists(path):
            m = keras.models.load_model(path)
            models.append(m)
            print(f"Loaded: {path}")
        else:
            print(f"Not found: {path}")
    return models
#=========================================================
# INPUTS
GMType_= ["Mix","PulseLike", "NoPulse"]
ModelVersion_= ['20261117', '20261117', '20261117']
Title_= ["Mix","Pulse-Like", "No-Pulse"]
# fd, Td, k1d, GMSetID, TM/T0, y0=DM/D0, TM, betaM, BM
XCol = np.array([0,1,2,6,7]) # columns for features
# XCol = np.array([1,7]) # columns for features
YCol = np.array([8]) # columns for target values (output)
n_splits= 10
import matplotlib.pyplot as plt
font = {'family' : 'Times New Roman',
        'size'   : 11}
plt.rc('font', **font)
fig, axs= plt.subplots(nrows=1, ncols=len(GMType_), figsize=(len(GMType_)*2.5,2.5))
for iGMType, GMType in enumerate(GMType_):
    ModelVersion= ModelVersion_[iGMType]
    #--------
    TestSetFile= f'../Data/BM_Test_{GMType}_HighDM.txt'
    DirOfSaveModels= f'ANN_Models/{GMType}/{str(len(XCol))}Features_{ModelVersion}'
    #========================================================
    # load scalers
    input_scaler= load(open(f"{DirOfSaveModels}/InputScaler.pkl", 'rb'))
    output_scaler= load(open(f"{DirOfSaveModels}/OutputScaler.pkl", 'rb'))
    # Load the models
    models= load_keras_ensemble(DirOfSaveModels,n_splits= n_splits)
    # load all data
    TestDat = np.loadtxt(TestSetFile)
    X_test = input_scaler.transform(TestDat[:,XCol])
    Y_test = TestDat[:,YCol]
    # make prediction
    pred_test = np.mean([m.predict(X_test) for m in models], axis=0)
    # Scaling the predicted data back to original output scale
    pred_test=output_scaler.inverse_transform(pred_test)
    #------------------------------------------------
    # Evaluate the prediction:
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.ravel(pred_test),np.ravel(Y_test))
    r2_Test = r_value**2
    print("R^2 correlation on the test set = " + str(r2_Test))
    #------------------------------------------------
    # Plot
    axs[iGMType].scatter(pred_test, Y_test,marker='.',s=4)
    axs[iGMType].text(1.5,0.75,'R2='+str(np.round(r2_Test,4)))
    axs[iGMType].plot(np.array([0.5,2.5]),np.array([0.5,2.5]),c='k',lw=1.5)
    axs[iGMType].grid()
    axs[iGMType].set_xlim(0.5,2.5)
    axs[iGMType].set_ylim(0.5,2.5)
    axs[iGMType].set_xticks([0.5,1.0,1.5,2,2.5])
    axs[iGMType].set_yticks([0.5,1.0,1.5,2,2.5])
    axs[iGMType].set_aspect('equal')
    axs[iGMType].set_xlabel('Predicted $B_M$')
    axs[iGMType].text(0.55,2.25,Title_[iGMType])
axs[0].set_ylabel('Actual $B_M$')

fig.tight_layout(pad=1.0)
plt.show()



