#Imports important libraries and custom functions
from splitDataset import split
from loadDataset import loadDataset
from models import  CNNGRUAE

#Length of the input sequance
inputSequance=8
#Length of the output sequance (Prediction)
outputSequance=4
#Length of the prediction step for Matplotlib visualization
predStep=2880
#Pre- and Post-COVID dataset paths
PreCovid='Dataset/EC_Pre-Covid_Data(March-2017).csv'
PostCovid='Dataset/EC_Post-Covid_Data(March-2020).csv'
#Load and preprocess data 
PreCovidData, PreCovidlabels, PreCovidlabelsMaxValues, scaler=loadDataset(PreCovid, inputSequance)
PostCovidData, PostCovidlabels, PostCovidlabelsMaxValues, scaler=loadDataset(PostCovid, inputSequance)
#Input and output sequance generation
PreCovidData, PreCovidlabels=split(PreCovidData, PreCovidlabels, PreCovidlabelsMaxValues,inputSequance,outputSequance)
PostCovidData, PostCovidlabels=split(PostCovidData, PostCovidlabels, PostCovidlabelsMaxValues,inputSequance,outputSequance)
#Number of epochs for the training
epochs=1
#Propposed model for load prediciton
CNNGRUAE(PreCovidData, PreCovidlabels, PostCovidData, PostCovidlabels, predStep, scaler, epochs)