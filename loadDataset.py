#Import important libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#Maximum Values Selection from whole dataset
def maxValuesSelection(TotalLength, inputSequance):
    c=1
    for i in range (TotalLength):
        
        
        lenth=TotalLength/inputSequance
        TotalLengtH= TotalLength-1
        if TotalLength%inputSequance==0:
            break
    return TotalLengtH
#Load and preprocess the data
def loadDataset(DSName, inputSequance):
    dataset=pd.read_csv(DSName, header=0)
    Length=maxValuesSelection(len(dataset), inputSequance)
    dataset=dataset.interpolate(method='linear',window=3)
    dataset=dataset.fillna(1)
    label=dataset.Global_active_power
    label=label.values
    data=dataset.values
    data=data[:, 2:7]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    label=label.reshape(-1, 1)
    label= scaler.fit_transform(label)
    label=np.array(label).reshape(-1)
    return data, label, Length, scaler
    
