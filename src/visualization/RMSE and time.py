import matplotlib.pyplot as plt
import math

tabPFN_error = [2.45737753060368, 4.315432019859161, 1.960744659352225, 5.5725002327313105, 4.9746004675174085]
tabPFN_time = [8140.9676197, 6565.0841218000005, 7638.175244499999, 5730.6826069, 5767.8530459]
tabPFN_time = [math.log(x) for x in tabPFN_time]

catboost_error = [3.4405386923567507, 4.862824295881633, 2.0414723432668413, 5.934675231172292, 5.507043302410141]
catboost_time = [145.49239859999943, 148.1466805, 146.31390459999966, 137.24440810000033,140.2006497000002]
catboost_time = [math.log(x) for x in catboost_time]

XGBoost_error = [3.4055097074510408, 4.714908120244955, 2.0697841768850975, 6.124260538254497, 5.767960177569115]
XGBoost_time = [16.928895099999863, 16.772931900000003, 16.26508829999966, 16.0611171999999, 16.129809800000658]
XGBoost_time = [math.log(x) for x in XGBoost_time]

mice_error = [3.3937282541939062, 4.716121215608436, 6.041635604273491, 5.422433854494632]
mice_time = [1.4444845999999991, 16.144688600000002, 2.807978200000001, 2.056051400000001]
mice_time = [math.log(x) for x in mice_time]

KNN_error = [12.158203793266749, 14.111119222356768,11.618133795327386, 13.61382034260977, 11.860019518870674]
KNN_time = [0.2184986000000002,0.2853338999999977,0.12342000000000297, 0.3252302, 0.28643539999999845]
KNN_time = [math.log(x) for x in KNN_time]

mean_error = [11.675853915247552, 12.955336819055825, 10.299083528664326, 12.71945147570113, 11.149979197852938]
mean_time = [0.006823899999996996, 0.02517870000000144, 0.015673800000001847, 0.010163800000000833, 0.00933729999999855]
mean_time = [math.log(x) for x in mean_time]

plt.scatter(tabPFN_error, tabPFN_time, label = 'tabPFN')
plt.scatter(catboost_error, catboost_time, label = 'Catboost')
plt.scatter(XGBoost_error, XGBoost_time, label = 'XGBoost')
plt.scatter(mice_error, mice_time, label='Mice')
plt.scatter(KNN_error, KNN_time, label='KNN')
plt.scatter(mean_error, mean_time, label='Mean')

plt.xlabel('RMSE')
plt.ylabel('Runtime in sec (log scale)')
plt.title('RMSE and Runtime, with the numbers as labels indicating the nan fraction in the dataset')
plt.annotate('0.1',(tabPFN_error[0], tabPFN_time[0]))
plt.annotate('0.1',(XGBoost_error[0], XGBoost_time[0]))
plt.annotate('0.1',(catboost_error[0], catboost_time[0]))
plt.annotate('0.1',(mice_error[0], mice_time[0]))
plt.annotate('0.1',(KNN_error[0], KNN_time[0]))
plt.annotate('0.1',(mean_error[0], mean_time[0]))

plt.annotate('0.15',(tabPFN_error[1], tabPFN_time[1]))
plt.annotate('0.15',(XGBoost_error[1], XGBoost_time[1]))
plt.annotate('0.15',(catboost_error[1], catboost_time[1]))
plt.annotate('0.15',(mice_error[1], mice_time[1]))
plt.annotate('0.15',(KNN_error[1], KNN_time[1]))
plt.annotate('0.15',(mean_error[1], mean_time[1]))

plt.annotate('0.05',(tabPFN_error[2], tabPFN_time[2]))
plt.annotate('0.05',(XGBoost_error[2], XGBoost_time[2]))
plt.annotate('0.05',(catboost_error[2], catboost_time[2]))
#plt.annotate('0.05',(mice_error[2], mice_time[2]))
plt.annotate('0.05',(KNN_error[2], KNN_time[2]))
plt.annotate('0.05',(mean_error[2], mean_time[2]))

plt.annotate('0.2',(tabPFN_error[3], tabPFN_time[3]))
plt.annotate('0.2',(XGBoost_error[3], XGBoost_time[3]))
plt.annotate('0.2',(catboost_error[3], catboost_time[3]))
plt.annotate('0.2',(mice_error[2], mice_time[2]))
plt.annotate('0.2',(KNN_error[3], KNN_time[3]))
plt.annotate('0.2',(mean_error[3], mean_time[3]))

plt.annotate('0.25',(tabPFN_error[4], tabPFN_time[4]))
plt.annotate('0.25',(XGBoost_error[4], XGBoost_time[4]))
plt.annotate('0.25',(catboost_error[4], catboost_time[4]))
plt.annotate('0.25',(mice_error[3], mice_time[3]))
plt.annotate('0.25',(KNN_error[4], KNN_time[4]))
plt.annotate('0.25',(mean_error[4], mean_time[4]))
plt.legend()


plt.savefig('test.png')







