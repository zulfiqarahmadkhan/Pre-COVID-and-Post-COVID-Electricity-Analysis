from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Conv1D
import keras as keras
from keras.layers import Flatten, GRU, Reshape,   MaxPooling1D, Dense, GlobalAveragePooling1D
import os
from plotdata import plot
from datetime import datetime
from keras_self_attention import SeqSelfAttention
from Losses import Prediction

random_seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
print("seed: ", random_seed)

def traintest(X1, Y1):
    X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.20, random_state=42)
    return X_train, y_train, X_test, y_test 
    


def CNNGRUAE(X1, Y1, PostCovidData, PostCovidlabels, predStep, scaler, epoch):
    
    #Initialize Network parameters
    namePre='precovid'
    namepost='postcovid'
    X, Y, testx, testy = traintest(X1, Y1)
    FilterSize1, FilterSize2= 32, 64
    kernel_size1, kernel_size2, kernel_size3 = 5, 3,1
    Activations= 'relu'
    batches = 4
    epochs = epoch
    CellSize1, CellSize2 = 32, 64
    num_outputs = 4
    learningRate = 0.001
    
    #Network archatecture
    inputs = keras.Input(shape=(8,5))
    CNN = Conv1D(filters=FilterSize1, kernel_size=kernel_size1, activation=Activations)(inputs)
    CNN = Conv1D(filters=FilterSize2, kernel_size=kernel_size2, activation=Activations)(CNN)
    CNN = Conv1D(filters=FilterSize2, kernel_size=kernel_size3, activation=Activations)(CNN)
    MP = MaxPooling1D(pool_size=2)(CNN)
    Seq = GRU(CellSize1, return_sequences=True)(MP)
    Seq = GRU(CellSize2, return_sequences=False)(Seq)
    Seq = Dense(64)(Seq)
    Seq = Flatten(name='Seq')(Seq)
    RS=Reshape((8, 8))(Seq)
    Seq = SeqSelfAttention(attention_activation='sigmoid')(RS)
    Seq=Reshape((8, 8))(Seq)
    FC = Dense(32)(Seq)
    FC = Flatten()(Seq)
    FCF = Dense(num_outputs)(FC)
    model = keras.Model(inputs=inputs, outputs=FCF, name="cnngru-ae")
    optimizer = keras.optimizers.Adam(learning_rate=learningRate)
    model.compile(loss="mse", optimizer=optimizer)
    model.summary()
    
    #Network training over precovid data 
    hist = model.fit(X, Y, epochs=epochs, batch_size=batches, verbose=1, validation_split=0.1) 
    
    #Network testing over precovid data
    predPre=model.predict(testx,verbose=1, batch_size=batches)
    testy = [x[0] for x in testy]
    predPre = [y[0] for y in predPre]
    testy=np.array(testy)
    predPre=np.array(predPre)
    '''actualvaluessPre=testx
    predictiosvaluesPre=predPre
    predPre=predPre.reshape(-1, 1)
    testy=testy.reshape(-1, 1)
    predPre=scaler.inverse_transform(predPre)
    testy=scaler.inverse_transform(testy)
    predPre=np.array(predPre).reshape(-1)
    testy=np.array(testy).reshape(-1)'''
    #plot precovid actual and predicted values, and find error rates
    plot(predPre, testy, predStep, namePre)
    Prediction(testy, predPre, namePre)
    
    #Network testing over postcovid data
    predPost=model.predict(PostCovidData,verbose=1, batch_size=batches)
    PostCovidlabels = [x[0] for x in PostCovidlabels]
    predPost = [y[0] for y in predPost]
    print('lenth', len(predPost), len(PostCovidlabels))
    predPost=np.array(predPost)
    PostCovidlabels=np.array(PostCovidlabels)
    '''actualvaluessPost=PostCovidlabels
    predictiosvaluesPost=predPost

    predPost=predPost.reshape(-1, 1)
    PostCovidlabels=PostCovidlabels.reshape(-1, 1)
    predPost=scaler.inverse_transform(predPost)
    PostCovidlabels=scaler.inverse_transform(PostCovidlabels)
    PostCovidlabels=np.array(PostCovidlabels).reshape(-1)
    predPost=np.array(predPost).reshape(-1)'''
    #plot precovid actual and predicted values, and find error rates
    plot(predPost, PostCovidlabels, predStep, namepost)
    Prediction(predPost, PostCovidlabels, namepost)







