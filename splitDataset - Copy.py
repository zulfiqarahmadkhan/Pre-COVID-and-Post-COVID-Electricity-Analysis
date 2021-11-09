from numpy import array
import numpy as np
def split(data, pred, max_values, inpSeq, outSeq):
    Y = []
    X=[]
    
    cnt=max_values/inpSeq
    cnt=int(cnt)
    c=0
    for i in range (cnt):
        
        X.append(data[c:c+inpSeq])
        Y.append(pred[c+inpSeq:c+inpSeq+outSeq])
        c+=1

    X=np.asarray(X)
    Y=np.asarray(Y)
    return X,Y
