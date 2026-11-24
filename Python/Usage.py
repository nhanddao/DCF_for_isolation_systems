# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 07:31:06 2026

@author: nhand
"""
import numpy as np
from tensorflow import keras
keras.utils.disable_interactive_logging()
from pickle import load
import os

def BM_ASCE7_22(beta_M=0.05):
    # Compute damping correction factor per ASCE 7-22
    # Parameters
    # - beta_M: damping ratio
    #-----------------
    beta=np.array([0.02,0.05,0.10,0.20,0.30,0.40,0.50])
    BM= np.array([0.8,1,1.2,1.5,1.7,1.9,2])
    if beta_M<= 0.02:
        return 0.8
    if beta_M>= 0.50:
        return 2.0
    return np.interp(beta_M,beta,BM)
def BM_New(beta_M=0.05, Td=4.0, GMType= "PulseLike"):
    # Calculate damping correction factor per the new model
    # Parameters
    # - beta_M= damping ratio
    # - Td= post-yield stiffness
    # - GMType= "PulseLike", "Mix", "NoPulse"
    if GMType== "Mix":
        c1,c2,c3,c4,c5,c6,c7,c8,c9= (-0.4302,	3.6815,	0.26816,	-11.069,	-0.02402,	-2.4434,	9.2937,	0.1726,	0.5337)
    if GMType== "PulseLike":
        c1,c2,c3,c4,c5,c6,c7,c8,c9= (0.45261,	-3.8211,	-0.085503,	4.6392,	0.0039788,	0.64268,	2.225,	-0.056449,	0.98454)
    if GMType== "NoPulse":
        c1,c2,c3,c4,c5,c6,c7,c8,c9= (-0.57723,	4.7446,	0.34017,	-13.653,	-0.032001,	-2.9365,	10.631,	0.22137,	0.44621)
    return (c1*beta_M**2*Td**2 + c2*beta_M**2*Td + c3*beta_M*Td**2+
            c4*beta_M**2 + c5*Td**2 + c6*beta_M*Td + c7*beta_M + c8*Td + c9)
def load_keras_ensemble(base_path,n_splits=10):
    models = []
    for i in range(1, n_splits+1):
        path = f"{base_path}/ANNModel_fold{i}.keras"
        if os.path.exists(path):
            m = keras.models.load_model(path)
            models.append(m)
        else:
            print(f"Not found: {path}")
    return models
def BM_ANN(f_d,T_d,k_1d,T_M,beta_M, GMType, ANNModels, input_scaler, output_scaler):
    # make prediction
    X = input_scaler.transform(np.array([[f_d,T_d,k_1d,T_M,beta_M]]))
    pred_test = np.mean([m.predict(X) for m in ANNModels], axis=0)
    # Scaling the predicted data back to original output scale
    pred_test=output_scaler.inverse_transform(pred_test)
    return pred_test[0][0]
def ELF(fd, Td, k1d, S1, BM_Model= "ASCE7_22", GMType= "PulseLike"):
    # Calculate the peak displacement per ELF in ASCE 7-22
    # return DM, TM, betaM (%)
    # parameters
    # - fd= normalized characteristic strength
    # - Td= post-yield period
    # - k1d= K1/Kd
    # - S1 (g)= spectral acceleration at 1 s
    # - BM_Model= "ASCE7_22", "New", "ANN"
    # - GMType= "PulseLike", "Mix", "NoPulse"
    g= 9.81
    Dy= g*Td**2*fd/4/np.pi**2/(k1d-1)
    Dold= 0
    DM= 0.5
    nIter= 0
    if BM_Model== "ANN":
        DirOfSaveModels= f'ANN_Models/{GMType}/5Features_20261117'
        # load scalers
        input_scaler= load(open(f"{DirOfSaveModels}/InputScaler.pkl", 'rb'))
        output_scaler= load(open(f"{DirOfSaveModels}/OutputScaler.pkl", 'rb'))
        # Load the models
        ANNModels= load_keras_ensemble(DirOfSaveModels)
        
    while abs(DM-Dold) > 0.0001*DM and nIter<=100:
        Dold= DM
        if DM >= Dy:
            betaM= 2*g*Td**2*fd*(DM-Dy)/np.pi/(4*np.pi**2*DM+g*Td**2*fd)/DM
            TM= 2*np.pi*(1/(4*np.pi**2/Td**2+g*fd/DM))**0.5
        else:
            betaM= 0
            TM= Td/k1d**0.5
        if BM_Model== "ASCE7_22":
            BM= BM_ASCE7_22(betaM)
        if BM_Model== "New":
            BM= BM_New(betaM, Td, GMType= GMType)
        if BM_Model== "ANN":
            BM= BM_ANN(fd,Td,k1d,TM,betaM, GMType, ANNModels,input_scaler,output_scaler)
        DM= g*S1*TM/4/np.pi**2/BM
        nIter= nIter+1
    return (DM, TM, betaM)
########################################################
if __name__=="__main__":
    fd= 0.05
    Td= 4
    k1d= 10
    S1= 0.7 # unit of g
    BM_Model= "New" # "New" for the proposed BM model; "ASCE7_22" for ASCE 7-22 BM model; "ANN" for ANN BM model
    GMType= "PulseLike"
    DM, TM, betaM= ELF(fd, Td, k1d, S1, BM_Model= BM_Model, GMType= GMType)
    print(f"DM= {DM}, TM= {TM}, betaM= {betaM}")