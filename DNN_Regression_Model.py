import pandas as pd
import pandas
import numpy
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import pickle
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# load data and arrange into Pandas dataframe
path = r'C:\Users\Shubham Jaiswal\Desktop\data analysis'
dff2=pd.read_csv(path + "\\ts_prediction_dataset.csv")
dff6=dff2[~(np.isnan(dff2["6_hr_50km"]))]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)

from sklearn.model_selection import train_test_split
pc=['PC' + str(x) for x in range(1, 13)]

train, test = train_test_split(dff6[pc+["utc","6_hr_50km"]], test_size=0.2,random_state=1)


#Scale data, otherwise model will fail.
#Standardize features by removing the mean and scaling to unit variance
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# scaler.fit(X_train)

# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)
X = train[pc]
y = train['6_hr_50km']

# define the model
#Experiment with deeper and wider networks
model = Sequential()
model.add(Dense(128, input_dim=12, activation='relu'))
model.add(Dense(64, activation='relu'))
#Output layer
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.summary()

model.fit(X, y,epochs =100)

history = model.fit(X, y, epochs =100)
###########################################
# acc = history.history['mean_absolute_error']
# val_acc = history.history['val_mean_absolute_error']
# plt.plot(epochs, acc, 'y', label='Training MAE')
# plt.plot(epochs, val_acc, 'r', label='Validation MAE')
# plt.title('Training and validation MAE')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

############################################
#Predict on test data
predicted_Prec =model.predict(test[pc])
predicted_Prec=predicted_Prec.reshape(449,)
test_y=test['6_hr_50km']
test_y1=test_y.to_numpy()
test_y1=test_y1.reshape(449,)
#.................................................
from sklearn.metrics import mean_squared_error
import math
import numpy

MSE = mean_squared_error(test_y,predicted_Prec)
 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)
CORR = np.corrcoef(test_y1,predicted_Prec)
print("Correlation_coeficient_is:\n")
print(CORR)
bias=np.mean(test_y1-predicted_Prec)
print(f"bias is:{bias}")
fig=plt.figure(figsize=(15,15))
plt.scatter(test_y,predicted_Prec)
plt.xlim([0,30])
plt.ylim([0,30])
# xpoints = np.array([1, 19])
# ypoints = np.array([1, 19])
xyz=numpy.polyfit(test_y1,predicted_Prec, 1)
f=np.poly1d(xyz)
lst=[0,50]
font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}
plt.title("DNN Regression", fontdict = font1)
plt.xlabel("Actual Precipitation", fontdict = font2)
plt.ylabel("Predicted Precipitation", fontdict = font2)

plt.plot(lst,f([0,50]),linewidth = '2',c='r')
xpoints = np.array([0, 30])
ypoints = np.array([0, 30])

plt.plot(xpoints, ypoints,linestyle = 'dashed',c="Black")
#plt.savefig("DNN_Regression.png",dpi=300)
plt.show()