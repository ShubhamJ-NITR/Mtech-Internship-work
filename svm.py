import pandas as pd
import pandas
import numpy
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import ensemble
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pickle
dff2=pd.read_csv("ts_prediction_dataset.csv")
dff6=dff2[~(np.isnan(dff2["6_hr_50km"]))]
# dff12=dff2[~(np.isnan(dff2["12_hr_50km"]))]
# dff18=dff2[~(np.isnan(dff2["18_hr_50km"]))]
# dff12=dff2[~(np.isnan(dff2["12_hr_50km"]))]
from sklearn.model_selection import train_test_split
pc=['PC' + str(x) for x in range(1, 13)]

train, test = train_test_split(dff6[pc+["utc","6_hr_50km"]], test_size=0.2,random_state=1)

#.....................................................
#multiple Regression model
import pandas
from sklearn import linear_model

#df = pandas.read_csv("cars.csv")

X = train[pc]
y = train['6_hr_50km']

# mod = linear_model.LinearRegression()
# mod=linear_model.LogisticRegression()
mod = svm.SVR()
mod.fit(X, y)

predicted_Prec = mod.predict(test[pc])
predicted_Prec1=np.delete(predicted_Prec,144)
test_y=test['6_hr_50km']
test_y2=test_y.to_numpy()
test_y1=np.delete(test_y2,266)

#............................................
from sklearn.metrics import mean_squared_error
import math
import numpy

MSE = mean_squared_error(test_y1,predicted_Prec1)
 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)
CORR = np.corrcoef(test_y1,predicted_Prec1)
print("Correlation_coeficient_is:\n")
print(CORR)
bias=np.mean(test_y1-predicted_Prec1)
print(f"bias is:{bias}")
fig=plt.figure(figsize=(15,15))
plt.scatter(test_y1,predicted_Prec1)
plt.xlim([0,30])
plt.ylim([0,30])
# xpoints = np.array([1, 19])
# ypoints = np.array([1, 19])
xyz=numpy.polyfit(test_y1,predicted_Prec1, 1)
f=np.poly1d(xyz)
lst=[0,50]
font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}
plt.title("SVM Regression", fontdict = font1)
plt.xlabel("Actual Precipitation", fontdict = font2)
plt.ylabel("Predicted Precipitation", fontdict = font2)

plt.plot(lst,f([0,50]),linewidth = '2',c='r')
xpoints = np.array([0, 30])
ypoints = np.array([0, 30])

plt.plot(xpoints, ypoints,linestyle = 'dashed',c="Black")
#plt.savefig("SVM_Regression.png",dpi=300)
plt.show()