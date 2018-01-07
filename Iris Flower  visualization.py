#Importing Package
import sys
print('Python: {}'.format(sys.version))
import numpy
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np

import warnings

#Reading IRIS Dataset in Pandas Dataframe
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#Printing DataFrame -only first 20 rows to understand what data look like
print(dataset.head(20))
#Output :Data have 5 Columns ,First four are features and fifth is Classfication of the Iris type


#findout unique classification/type of iris flower.
dataset['class'].unique()

print(dataset.describe())
#Output :count tells that all the 4 features have 150 rows. In general ,sepal is larger than petal as we can see from mean

#findout no of rows for each class.
print(dataset.groupby('class').size())

###########################################################################################################
#Create 3 DataFrame for each class
setosa=dataset[dataset['class']=='Iris-setosa']
versicolor =dataset[dataset['class']=='Iris-versicolor']
virginica =dataset[dataset['class']=='Iris-virginica']

print(setosa.describe())
print(versicolor.describe())
print(virginica.describe())
#Output :for setosa  ,sepal is a lot larger than petal as we can see from mean in comparison to other class
###########################################################################################################

#############################Plotting Petal Length vs Petal Width & Sepal Length vs Sepal width############
plt.figure()
fig,ax=plt.subplots(1,2,figsize=(17, 9))
dataset.plot(x="sepal-length",y="sepal-width",kind="scatter",ax=ax[0],sharex=False,sharey=False,Label="sepal",color='r')
dataset.plot(x="petal-length",y="petal-width",kind="scatter",ax=ax[1],sharex=False,sharey=False,Label="petal",color='b')
ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')
ax[1].set(title='Petal Comparasion',  ylabel='petal-width')
ax[0].legend()
ax[1].legend()
plt.show()
plt.close()
# we can see that for petal there are few petals which are smaller than rest of petal.Let's examine them
###########################################################################################################

###########################################################################################################
# #for each class ,let's check what is petal and sepal distibutuon
plt.figure()
fig,ax=plt.subplots(1,2,figsize=(17, 6))
setosa.plot(x="sepal-length", y="sepal-width", kind="scatter",ax=ax[0],Label='setosa',color='r')
versicolor.plot(x="sepal-length",y="sepal-width",kind="scatter",ax=ax[0],Label='versicolor',color='b')
virginica.plot(x="sepal-length", y="sepal-width", kind="scatter", ax=ax[0], Label='virginica', color='g')

setosa.plot(x="petal-length", y="petal-width", kind="scatter",ax=ax[1],Label='setosa',color='r')
versicolor.plot(x="petal-length",y="petal-width",kind="scatter",ax=ax[1],Label='versicolor',color='b')
virginica.plot(x="petal-length", y="petal-width", kind="scatter", ax=ax[1], Label='virginica', color='g')
ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')
ax[1].set(title='Petal Comparasion',  ylabel='petal-width')
ax[0].legend()
ax[1].legend()
plt.show()
plt.close()
#satosa   - satosa Petal are relatively smaller than rest of species .can be easily separable from rest of Species
#versicolor & virginica are also separable in Petal comprasion
#satoa sepal are smallest in length and largest in Width than other species
###########################################################################################################

###############################Histogram# all feature  for all class###########################################################################
plt.figure()

fig,ax=plt.subplots(4,3,figsize=(17, 8))
setosa["sepal-length"].plot(kind="hist", ax=ax[0][0],label="setosa",color ='r',fontsize=10)
versicolor["sepal-length"].plot(kind="hist", ax=ax[0][1],label="versicolor",color='b',fontsize=10)
virginica["sepal-length"].plot( kind="hist",ax=ax[0][2],label="virginica",color='g',fontsize=10)

setosa["sepal-width"].plot(kind="hist", ax=ax[1][0],label="setosa",color ='r',fontsize=10)
versicolor["sepal-width"].plot(kind="hist", ax=ax[1][1],label="versicolor",color='b',fontsize=10)
virginica["sepal-width"].plot( kind="hist",ax=ax[1][2],label="virginica",color='g',fontsize=10)

setosa["petal-length"].plot(kind="hist", ax=ax[2][0],label="setosa",color ='r',fontsize=10)
versicolor["petal-length"].plot(kind="hist", ax=ax[2][1],label="versicolor",color='b',fontsize=10)
virginica["petal-length"].plot( kind="hist",ax=ax[2][2],label="virginica",color='g',fontsize=10)


setosa["petal-width"].plot(kind="hist", ax=ax[3][0],label="setosa",color ='r',fontsize=10)
versicolor["petal-width"].plot(kind="hist", ax=ax[3][1],label="versicolor",color='b',fontsize=10)
virginica["petal-width"].plot( kind="hist",ax=ax[3][2],label="virginica",color='g',fontsize=10)

plt.rcParams.update({'font.size': 10})
plt.tight_layout()

ax[0][0].set(title='sepal-length')
ax[0][1].set(title='sepal-length')
ax[0][2].set(title='sepal-length')
ax[1][0].set(title='sepal-width ')
ax[1][1].set(title='sepal-width ')
ax[1][2].set(title='sepal-width ')
ax[2][0].set(title='petal-length')
ax[2][1].set(title='petal-length ')
ax[2][2].set(title='petal-length')
ax[3][0].set(title='petal-width')
ax[3][1].set(title='petal-width')
ax[3][2].set(title='petal-width')

ax[0][0].legend()
ax[0][1].legend()
ax[0][2].legend()
ax[1][0].legend()
ax[1][1].legend()
ax[1][2].legend()
ax[2][0].legend()
ax[2][1].legend()
ax[2][2].legend()
ax[3][0].legend()
ax[3][1].legend()
ax[3][2].legend()

plt.show()
plt.close()
###########################################################################################################