import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

data1=pd.read_csv("RSRW_Indices.csv")
print(data1.columns)
data2=data1.drop(['year', 'month', 'day', 'utc', 'time_epoch', 'sta_id', 'lat', 'lon', 'ele',
       'ws_850_500', 'ws_500_200', 'ws_850_200', 'waa', 'date', 'caa',
       'ws_950_700'],axis="columns")
print(data2.columns)
#print(data2.isnull().sum())
data=data2.dropna()
#########################
#
# Perform PCA on the data
#
#########################
# First center and scale the data
scaled_data = preprocessing.scale(data) 
pca = PCA() # create a PCA object
pca.fit(scaled_data) # do the math
pca_data = pca.transform(scaled_data) # get PCA coordinates for scaled_data
 
#########################
#
# Draw a scree plot
#
#########################
 
#The following code constructs the Scree plot
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['pc' + str(x) for x in range(1, len(per_var)+1)]
fig=plt.figure(figsize=(15,10))
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels,width=.6)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
#plt.savefig("Scree_plot.png",dpi=300)
#compp=pca.components_

# import pickle

# pickle.dump(pca,open('pca.pkl','wb'))
# pca=pickle.load(open('pca.pkl','rb'))
# df.rename(columns={'1':"PC1"},inplace=True)
# pc_name={str(n-1):"PC"+str(n) for n in range(1,13)}
