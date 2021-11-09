from numpy import array
import pandas as pd

def loadDataset(DSName):
    

    '''datecols=['timestamp']
    edata = pd.read_csv(DSName,  parse_dates=datecols, header=0)
    
    edata.timestamp=pd.to_datetime(edata.timestamp, utc= True)
    start_date7 = pd.to_datetime('6/1/2014 11:00:00 AM', utc= True)
    end_date7 = pd.to_datetime('6/14/2016 11:59:00 PM', utc= True)
    dataset=edata.loc[((edata['timestamp'] > start_date7) & (edata['timestamp'] < end_date7))]
    print(dataset)'''

    dataset=pd.read_csv(DSName, header=0)
    dataset=dataset.interpolate(method='linear',window=3)
    dataset=dataset.fillna(1)
    label=dataset.Global_active_power
    print(label.shape)
    label=label.values
    data=dataset.values
    data=data[:, 1:8]



    return data, label
    
