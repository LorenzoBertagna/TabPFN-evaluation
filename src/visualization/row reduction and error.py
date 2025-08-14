from matplotlib import pyplot as plt

'''
Here the plot variables of the file Row analysis -> row_analysis.txt
train_fraction = [1, 0.5, 0.4, 0.3, 0.2, 0.1]
train_fraction_tabPFN = [0.5, 0.4, 0.3, 0.2, 0.1]


tabPFN_error = [3.7702244513179854, 3.6429681976021233, 4.066816911806256, 4.09215434325617,  4.065201380017159]
mice_error = [4.6227231119650325, 4.400002468714586, 3.796697452759771, 4.4122526180712995, 5.654700065163061,
              5.225656201308401]
KNN_error = [12.213214245531205,18.33199481392475,12.170406897606695,12.793357237120514, 13.605926358414571,
             13.166790880034947]
XGB_error = [3.621370283085703, 4.264012217098532, 4.55582983010456, 4.793285803251334, 4.258131457993262,
             5.633112600000459]
cat_error = [3.502779764692629, 3.917561233158931, 4.299768288108988, 4.403404688964862, 4.939917405066009,
             5.8521481062730505]
mean_error = [11.988730733586381, 12.019993649608372, 11.985940689250025, 11.989020242801137, 11.989185271963578,
              11.970787913128705]

tabPFN_time = [8259.6634218, 5272.8763524000005, 3542.598769, 1496.622978200001, 564.9419809999999]
'''

# Here the plot variables of Row Analysis -> row_analysis.txt
train_fraction = [ 0.1, 0.05, 0.02, 0.01]


tabPFN_error = [3.185802486765015,3.4485828951524784, 3.7377604041138155, 4.083127835704455]
mice_error = [7.23773095094774, 7.3866926213528705, 12.665978321812576, 11.30157490832905]
KNN_error = [12.826893402496506, 12.43421316828373, 15.69960716685129, 13.64315391127935]
XGB_error = [4.726165686396783, 5.504439922992927, 7.652620902803789, 11.129971374094522]
cat_error = [5.8347610844737385, 6.566629411621659, 8.49939870116791, 8.86518081490935]
mean_error = [11.465575013708085, 11.453233204038119, 12.906688582355043, 11.62044060379818]

tabPFN_time = [854.8987265, 378.75396939999996, 302.02096099999994, 191.91700319999995]
plt.plot(train_fraction, tabPFN_error, label='tabPFN', marker='s')
plt.plot(train_fraction, mice_error, label='mice', marker='s')
plt.plot(train_fraction, KNN_error, label='KNN', marker='s')
plt.plot(train_fraction, XGB_error, label='XGBoost', marker='s')
plt.plot(train_fraction, cat_error, label='Catboost', marker='s')
plt.plot(train_fraction, mean_error, label='mean', marker='s')
plt.legend()
plt.title('RMSE for different methods, depending on fraction of train data used')
plt.xlabel('Train fraction')
plt.ylabel('RMSE')
plt.show()

plt.plot(train_fraction, tabPFN_time, marker='s', color='b')
plt.xlabel('Train fraction')
plt.ylabel('Time in sec')
plt.title('Runtime of tabPFN for different fraction of train data')
plt.show()


plt.plot(train_fraction, tabPFN_error, label='tabPFN', marker='s')
plt.plot(train_fraction, mice_error, label='mice', marker='s')
plt.plot(train_fraction, XGB_error, label='XGBoost', marker='s')
plt.plot(train_fraction, cat_error, label='Catboost', marker='s')
plt.legend()
plt.title('RMSE for different methods, depending on fraction of train data used - No mean and KNN')
plt.xlabel('Train fraction')
plt.ylabel('RMSE')
plt.show()

