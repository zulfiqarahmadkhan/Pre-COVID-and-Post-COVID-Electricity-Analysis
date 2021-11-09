import matplotlib.pyplot as plt
import csv 

def plot(ypred,ytrue,predStep, name):
    plt.figure(figsize=(12, 6))
    plt.plot(ytrue[0:predStep],  color='royalblue', linewidth=1, label='actual')
    plt.plot(ypred[0:predStep], color='darkorange',  linewidth=1, label='predicted')
    plt.xlabel(' Time ', fontsize=11, color='black', style='normal', fontname="Time New Roman")
    plt.ylabel('Electricity consumption (KWh)', fontsize=10, color='black', style='normal', fontname="Time New Roman")
    plt.xticks(fontsize=10, color='black', style='normal', fontname="Time New Roman", ha="center")
    plt.yticks(fontsize=10, color='black', style='normal', fontname="Time New Roman", )
    plt.grid(True)
    plt.legend(fontsize=14, loc='upper right')
    plt.savefig('Results/'+name+'.png', bbox_inches='tight')



