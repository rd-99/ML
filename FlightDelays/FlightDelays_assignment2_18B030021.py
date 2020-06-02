import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
sns.set(style="white")
sns.set(rc={'figure.figsize':(15,5)})
sns.set(style="whitegrid", color_codes=True)
dataset=pd.read_csv("FlightDelays.csv")

flightstatus=dataset.iloc[:,-1]
crs_dep_time=dataset.iloc[:,0]
carrier=dataset.iloc[:,1]
dep_time=dataset.iloc[:,2]
dest=dataset.iloc[:,3]
distance=dataset.iloc[:,4]
fl_date=dataset.iloc[:,5]
fl_num=dataset.iloc[:,6]
origin=dataset.iloc[:,7]
weather=dataset.iloc[:,8]
day_week=dataset.iloc[:,9]
day_of_month=dataset.iloc[:,10]
tail_no=dataset.iloc[:,11]

#  ***Question 1***
#pd.crosstab(carrier,flightstatus).plot(kind='bar')
#plt.title('delayed status vs carrier')
#plt.xlabel('carrier')
#plt.ylabel('No. of Flights')
#plt.savefig('delayed status')
# 
#pd.crosstab(dest,flightstatus).plot(kind='bar')
#plt.title('delayed status vs destination')
#plt.xlabel('destination')
#plt.ylabel('No. of Flights')
#plt.savefig('delayed status')
# 
#pd.crosstab(distance,flightstatus).plot(kind='bar')
#plt.title('delayed status vs distance')
#plt.xlabel('distance')
#plt.ylabel('No. of Flights')
#plt.savefig('delayed status')
# 
#plt.figure(figsize=(59,50))
#pd.crosstab(fl_date,flightstatus).plot(kind='bar')
#plt.title('delayed status vs fl_date')
#plt.xlabel('fl_date')
#plt.ylabel('No. of Flights')
#plt.savefig('delayed status')
# 
#pd.crosstab(origin,flightstatus).plot(kind='bar')
#plt.title('delayed status vs airport origin')
#plt.xlabel('origin')
#plt.ylabel('No. of Flights')
#plt.savefig('delayed status')
# 
#plt.figure(figsize=(40,5))
#pd.crosstab(weather,flightstatus).plot(kind='bar')
#plt.title('delayed status vs weather')
#plt.xlabel('weather')
#plt.ylabel('No. of Flights')
#plt.savefig('delayed status')
# 
#pd.crosstab(day_week,flightstatus).plot(kind='bar')
#plt.title('delayed status vs day_of_week')
#plt.xlabel('day_of _the_week')
#plt.ylabel('No. of Flights')
#plt.savefig('delayed status')
# 
#plt.figure(figsize=(40,5))
#pd.crosstab(day_of_month,flightstatus).plot(kind='bar')
#plt.title('delayed status vs day_of_month')
#plt.xlabel('day_of_month')
#plt.ylabel('No. of Flights')
#plt.savefig('delayed status')
#plt.show()
#timerange=(500,2200)
#bins=18
#plots=np.ones(len(flightstatus))*(flightstatus=="delayed")*dep_time
#plt.hist(plots,bins,timerange,color='blue',histtype='bar',rwidth=0.75)
#plt.xlabel("Scheduled Departure Time")
#plt.ylabel("Number of flights delayed")
#plt.title("Variation with Scheduled Departure time")
#plt.show()
 



#   *** Question 2***

dataset.isnull().sum()  #no values are missing

def standard(data):
    mean=sum(data)/len(data)
    variance=sum((data-mean)**2)/len(data)
    z=(data-mean)/((variance)**0.5)
    return z

carrier1=np.ones(len(carrier))
temp=np.ones(len(carrier))
for i in range(len(carrier.unique())):
    carrier1=carrier1+(temp*(carrier==carrier.unique()[i])*i)
carrier1=standard(carrier1)

origin1=np.ones(len(origin))
temp=np.ones(len(origin))
for i in range(len(origin.unique())):
    origin1=origin1+(temp*(origin==origin.unique()[i])*i)
origin1=standard(origin1)

dest1=np.ones(len(dest))
temp=np.ones(len(dest))
for i in range(len(dest.unique())):
    dest1=dest1+(temp*(dest==dest.unique()[i])*i)
dest1=standard(dest1)

dayofweek1=np.ones(len(day_week))
temp=np.ones(len(day_week))
for i in range(len(day_week.unique())):
    dayofweek1=dayofweek1+(temp*(day_week==day_week.unique()[i])*i)
dayofweek1=standard(dayofweek1)

dayofmonth1=np.ones(len(day_of_month))
temp=np.ones(len(day_of_month))
for i in range(len(day_of_month.unique())):
    dayofmonth1=dayofmonth1+(temp*(day_of_month==day_of_month.unique()[i])*i)
dayofmonth1=standard(dayofmonth1)


fl_date1=np.ones(len(fl_date))
temp=np.ones(len(fl_date))
for i in range(len(fl_date.unique())):
    fl_date1=fl_date1+(temp*(fl_date==fl_date.unique()[i])*i)
fl_date1=standard(fl_date1)

fl_tail_no1=np.ones(len(tail_no))
temp=np.ones(len(tail_no))
for i in range(len(tail_no.unique())):
    fl_tail_no1=fl_tail_no1+(temp*(tail_no==tail_no.unique()[i])*i)
fl_tail_no1=standard(fl_tail_no1)


fl_num1=np.ones(len(fl_num))
temp=np.ones(len(fl_num))
for i in range(len(fl_num.unique())):
    fl_num1=fl_num1+(temp*(fl_num==fl_num.unique()[i])*i)
fl_num1=standard(fl_num1)

dist1=standard(distance)
dep_time1=standard(dep_time)
crs_dep_time1=standard(crs_dep_time)
flightstatus1=(np.ones(len(flightstatus)))*(flightstatus=="ontime")


intercept=pd.DataFrame(np.ones(len(flightstatus)),columns=["INTERCEPT"])
X=pd.concat((intercept,crs_dep_time1,carrier1,dep_time1,dest1,dist1,fl_date1,fl_num1,origin1,weather,dayofweek1,dayofmonth1,fl_tail_no1),axis=1)
Y=flightstatus1
X1=np.array(X)
Y1=np.array(Y)




from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size =4/10, random_state = 0)
def sigmoid_func(coeff,X):
      z=np.dot(X,coeff.T)
      np.dot(X,coeff.T)
      return 1.0/(1.0+np.exp(-z))
 
def grad(coeff,X,Y):
     first=sigmoid_func(coeff,X)-Y.reshape(X.shape[0],-1)
     final=np.dot(first.T,X)
     return final
     
def cost_function(coeff,X,Y):
     step1=Y*np.log(sigmoid_func(coeff,X))
     step2=(1-Y)*np.log(1-sigmoid_func(coeff,X))
     final=step1+step2
     return np.mean(-final)
 
def gradient_descent(X,Y,coeff,alpha=0.001,iter=5000):
     for i in range(iter):
         coeff=coeff-(alpha*grad(coeff,X,Y))
     return coeff
 
def pred_values(coeff,X):
     pred_prob=sigmoid_func(coeff,X)
     pred_value=np.where(pred_prob>=0.5,1,0)
     return np.squeeze(pred_value)
 
    
    
 
##########Without Feature Selection################################
coeff=np.matrix(np.zeros(X_train.shape[1]))
coeff=gradient_descent(X_train,Y_train,coeff)
MSE=np.sqrt(sum(((Y_train-pred_values(coeff,X_train))**2))/(len(X)-len(X.columns)))
se=np.sqrt(MSE*(np.linalg.inv(np.dot(X_train.T,X_train)).diagonal()))
T_score=np.divide(coeff,se)
p_values=[2*(1-stats.t.cdf(np.abs(i),len(X)-1)) for i in T_score]
se=np.round(se,3)
T_score=np.round(T_score,3)
coefnew=np.array([i for i in coeff])
print(coefnew[0][0].shape,se.shape,T_score[0].shape,p_values[0][0].shape)
df1=pd.DataFrame()
df1["Coefficients"],df1["Standard Errors"],df1["T Values"],df1["P Values"]=[coefnew[0][0],se,T_score[0],p_values[0][0]]
df1.index=X.columns
print(df1)
Ypred=pred_values(coeff,X_test)
print("Correctly predicted labels:",np.sum(Y_test==Ypred)) 
TP=0
TN=0
FP=0
FN=0
for i in range(len(Y_test)):
    if((Ypred[i]==1)and(Y_test[i]==1)):
        TP+=1
    elif((Ypred[i]==0)and(Y_test[i]==0)):
         TN+=1
    elif((Ypred[i]==1)and(Y_test[i]==0)):
         FN+=1
    else:
         FP+=1
total=(TP+TN+FN+FP)
print("Accuracy:",((TP+TN)/total)*100,"%")
print("Misclassification Rate(Error Rate):",((FN+FP)/total*100),"%")
print("True Positive Rate(Recall):",TP/(TP+FN)*100,"%")
print("False Positive Rate:",FP/(TN+FP)*100,"%")
print("True Negativity Rate(Specificity):",TN/(TN+FP)*100,"%")
print("Precision:",TP/(TP+FP)*100,"%")
print("Prevalence:",(TP+FN)/total*100,"%")


print(pd.DataFrame.corr(X))

##########CRS_DEP_TIME & DEP_TIME are almost same
##########FL_DATE & DAY_OF_MONTH are almost same
###########Feature Selection##############################
#FL_date,dep_time is removed as p-value>5%
tuple1=(0,1,2,4,5,7,8,9,10,11,12)
Xnew_train=X_train[:,tuple1]
Xnew_test=X_test[:,tuple1]
coeff=np.matrix(np.zeros(Xnew_train.shape[1]))
coeff=gradient_descent(Xnew_train,Y_train,coeff)
MSE=np.sqrt(sum(((Y_train-pred_values(coeff,Xnew_train))**2))/((Xnew_train.shape[0])-(Xnew_train.shape[1])))
se=np.sqrt(MSE*(np.linalg.inv(np.dot(Xnew_train.T,Xnew_train)).diagonal()))
T_score=np.divide(coeff,se)
p_value=[2*(1-stats.t.cdf(np.abs(i),len(X)-1)) for i in T_score]
se=np.round(se,4)
T_score=np.round(T_score,4)
coef_new=np.array([i for i in coeff])
print(coef_new[0][0].shape,se.shape,T_score[0].shape,p_value[0][0].shape)
df1=pd.DataFrame()
df1["Coefficients"],df1["Standard Errors"],df1["T Values"],df1["P Values"]=[coef_new[0][0],se,T_score[0],p_value[0][0]]
print(df1)
Ypred=pred_values(coeff,Xnew_test)
print("Labels predicted correctly:",np.sum(Y_test==Ypred))
#from scipy.stats import spearmanr
#coorelation=spearmanr(Xnew_test, Y_test)
#coorelation=np.round(coorelation,4)
#print(coorelation)
print("ttttttttttttttttttttttttttttttt")

print(X)
print("ttttttttttttttttttttttttttttttt")
print(X_train)
print("ttttttttttttttttttttttttttttttt")

print(X1)
print("ttttttttttttttttttttttttttttttt")


p_coef,p_value=stats.pearsonr(X.CARRIER,Ypred)
print(p_coef)
#########feature selection 2nd iteration#######################################
##new iteration
#tuple2=(0,1,3,6,8,9,11,12)
#Xnew_train=X_train[:,tuple2]
#Xnew_test=X_test[:,tuple2]
#
#coeff=np.matrix(np.zeros(Xnew_train.shape[1]))
#coeff=gradient_descent(Xnew_train,Y_train,coeff)
#MSE=np.sqrt(sum(((Y_train-pred_values(coeff,Xnew_train))**2))/((Xnew_train.shape[0])-(Xnew_train.shape[1])))
#se=np.sqrt(MSE*(np.linalg.inv(np.dot(Xnew_train.T,Xnew_train)).diagonal()))
#T_score=np.divide(coeff,se)
#p_value=[2*(1-stats.t.cdf(np.abs(i),len(X)-1)) for i in T_score]
#se=np.round(se,4)
#T_score=np.round(T_score,4)
#coef_new=np.array([i for i in coeff])
#print(coef_new[0][0].shape,se.shape,T_score[0].shape,p_value[0][0].shape)
#df1=pd.DataFrame()
#df1["Coefficients"],df1["Standard Errors"],df1["T Values"],df1["P Values"]=[coef_new[0][0],se,T_score[0],p_value[0][0]]
#print(df1)
#Ypred=pred_values(coeff,Xnew_test)
#print("Correctly predicted labels:",np.sum(Y_test==Ypred))
#TP=0
#TN=0
#FP=0
#FN=0
#for i in range(len(Y_test)):
#    if((Ypred[i]==1)and(Y_test[i]==1)):
#        TP+=1
#    elif((Ypred[i]==0)and(Y_test[i]==0)):
#        TN+=1
#    elif((Ypred[i]==1)and(Y_test[i]==0)):
#        FN+=1
#    else:
#        FP+=1
#total=(TP+TN+FN+FP)
#print("Accuracy:",((TP+TN)/total)*100,"%")
#print("Misclassification Rate(Error Rate):",((FN+FP)/total*100),"%")
#print("True Positive Rate(Recall):",TP/(TP+FN)*100,"%")
#print("False Positive Rate:",FP/(TN+FP)*100,"%")
#print("True Negativity Rate(Specificity):",TN/(TN+FP)*100,"%")
#print("Precision:",TP/(TP+FP)*100,"%")
#print("Prevalence:",(TP+FN)/total*100,"%")
#
#
#
