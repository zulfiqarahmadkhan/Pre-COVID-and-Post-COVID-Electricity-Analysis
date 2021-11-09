from math import sqrt
import numpy as np
import csv 
from sklearn.metrics import mean_squared_error, mean_absolute_error

def mape(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

def Prediction(actualvaluess, predictiosvalues, name):
	MSE = mean_squared_error(actualvaluess, predictiosvalues)
	MSE = round(MSE, 4)
	RMSE = sqrt(mean_squared_error(actualvaluess, predictiosvalues))
	RMSE = round(RMSE, 4)
	MAPE = mape(actualvaluess, predictiosvalues,)
	MAPE = round(MAPE, 4)
	header=['model', 'MSE',   'RMSE',  'MAPE']
	data =[name, MSE,  RMSE, MAPE]
	print('CNNGRUAE performance over: ', name, '\n', 'MSE: ', MSE,  'RMSE: ', RMSE, '\n', 'MAPE: ', MAPE)
	with open('Results/results.csv', 'a', newline='') as f:
		writer = csv.writer(f)
		# write the header
		writer.writerow(header)
		# write the data
		writer.writerow(data)
	
