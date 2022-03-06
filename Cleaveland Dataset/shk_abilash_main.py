#------------------------------------------------------------------------------------------------------------------------
"IMPORT THE LIBRARIES"

import pandas as pd
import numpy as np
import keras
from sklearn import preprocessing
import sweetviz
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 
from sklearn.model_selection import GridSearchCV
import seaborn as sns

from keras.models import Model
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers.merge import concatenate

#------------------------------------------------------------------------------------------------------------------------
"EXCLAMATORY DATA ANALYSIS"

# READ THE DATASET
df = pd.read_csv("Dataset/heart.csv")

# SHOWS THE ANALYSIS REPORT
report = sweetviz.analyze([df,'df'],target_feat='target')
report.show_html('cleaveland_report.html')

#------------------------------------------------------------------------------------------------------------------------
"SPLITTING INTO X AND Y"

x = df.drop("target",axis=1)
y = df["target"]

#Convert lists to arrays        
data_var = np.array(x,dtype=np.int)
data_labels = np.array(y)


#------------------------------------------------------------------------------------------------------------------------
"FEATURE SELECTION"


from Algorithm.genetic_sine import jfs 


xtrain, xtest, ytrain, ytest = train_test_split(data_var, data_labels, test_size=0.3)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

# parameter
k    = 5     # k-value in KNN
N    = 100    # number of particles
T    = 50   # maximum number of iterations
w    = 0.9
c1   = 10
c2   = 10
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'w':w, 'c1':c1, 'c2':c2}

# perform feature selection
fmdl = jfs(data_var, data_labels, opts)
sf   = fmdl['sf']

input_size = len(sf)
x_var  = data_var[:, sf]


#------------------------------------------------------------------------------------------------------------------------
"DATA NORMALIZATION FOR THE FEATURE EXTRACTION"

# Data reshaping for the Deep CNN
k = x_var.reshape(1, -1).shape

#Encode labels from text to integers.
le = preprocessing.LabelEncoder()
le.fit(y)
data_labels_encoded = le.transform(y)

y_reshaped = np.array(keras.utils.to_categorical(data_labels_encoded, 2))


#------------------------------------------------------------------------------------------------------------------------
"DEEP CNN FOR FEATURE EXTRACTION"


# MODEL 1
inp =  Input(shape=(input_size,1))
conv = Conv1D(filters=2, kernel_size=2,activation="relu")(inp)
pool = MaxPool1D(pool_size=2)(conv)
conv = Conv1D(filters=5, kernel_size=2,activation="relu")(inp)
pool = MaxPool1D(pool_size=2)(conv)
flat = Flatten()(pool)
dense = Dense(100)(flat)
model = Model(inp, dense)
model.compile(loss='mse', optimizer='adam')

print(model.summary())

# get some data
X=np.expand_dims(x_var, axis=2)
Y=np.expand_dims(data_labels,axis=1)

# fit model
model.fit(X, Y,epochs=500,batch_size=50,verbose=1)

# predict the features
conv1_predicted = model.predict(X)


#------------------------------------------------------------------------------------------------------------------------
"FEATURE EXTRACTION Inception V3"

" Inception V3 Model Feature Extraction"

def naive_inception_module(layer_in, f1, f2, f3):
	# 1x1 conv
	conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
	# 3x3 conv
	conv3 = Conv2D(f2, (3,3), padding='same', activation='relu')(layer_in)
	# 5x5 conv
	conv5 = Conv2D(f3, (5,5), padding='same', activation='relu')(layer_in)
	# 3x3 max pooling
	pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
	# concatenate filters, assumes filters/channels last
	layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out
 
# define model input
visible = Input(shape=(303, input_size, 1))
# add inception module
layer = naive_inception_module(visible, 21, 24, 54)
# create model
incept_model = Model(inputs=visible, outputs=layer)
incept_model.compile(loss='mse', optimizer='adam')
# summarize model
incept_model.summary()

incept_model.fit(X, Y,epochs=100,batch_size=50,verbose=1)

# Getting Weights from the learned NN
weights = [incept_model.get_weights() for layer in incept_model.layers]

# Model Prediction(Feature Extraction)
incept_pred = model.predict(X)

scal_features = np.concatenate((incept_pred,conv1_predicted),axis=1)

#------------------------------------------------------------------------------------------------------------------------
"SCATTER PCA - DIMENSIONALITY REDUCTION"

def Scatter_PCA(X , num_components):
     
    # Step-1
    X_meaned = X - np.mean(X , axis = 0)
     
    # Step-2
    cov_mat = np.cov(X_meaned , rowvar = False)
     
    # Step-3
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    # Step-4 
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    
    #Scatter Points Calculation 
    m,n = X.shape
    G = np.dot(X.T,X)
    D = np.zeros((n,n))
    for i in range(n):
       for j in range(i+1,n):
           D[i,j] = G[i,j] - 2*G[i,j] + G[j,j]
           D[j,i] = D[i,j]

    #Step-5
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
    #Step-6 
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
    
    #Step-7 
    eigenvector_subset = np.dot(D.transpose() , X_meaned.transpose() ).transpose()
     
     
    return X_reduced,eigenvector_subset

# Applying it to K Rank PCA function
mat_reduced,feat = Scatter_PCA(scal_features ,150)
mat_reduced.shape
# np.save("cleave_reduced.npy",mat_reduced)
# np.save("cleave_y.npy",Y)

# Saving the Reduced Features
x_value = np.load("cleave_reduced.npy")
y_value = np.load("cleave_y.npy")


#------------------------------------------------------------------------------------------------------------------------
"TRAIN TEST SPLIT"


x_train, x_test, y_train, y_test = train_test_split(x_value, y_value, test_size=0.1, random_state=1) # 70% training and 30% test

#------------------------------------------------------------------------------------------------------------------------
"GRID SEARCH WEIGHTED DECISION TREE"

# Model Declaration
clf = DecisionTreeClassifier()

acc= []

for i in range(len(weights[0][0][0][0][0])):
  clf.fit(x_train, y_train,sample_weight=weights[0][0][0][0][0][i])
  y_preds = clf.predict(x_value)
  print("\n\tACCURACY SCORE\n\t******************************\n")
  print (f"\t{metrics.accuracy_score(y_preds, y_value)*100}")
  acc.append(metrics.accuracy_score(y_preds, y_value)*100)


"GRID SEARCH CV METRICS"
parameters = {'criterion':('gini', 'entropy'), 'max_depth':[10, 20,50,100,150]}
clf_grid= GridSearchCV(clf, parameters)
clf_grid.fit(x_train, y_train)

y_preds_grid = clf_grid.predict(x_value)

# Saving the Predicted Values
# np.save("y.npy",y_value)
# np.save("y_value.npy",y_preds_grid)

y = np.load("y.npy")
y_preds = np.load("y_value.npy")


#------------------------------------------------------------------------------------------------------------------------
"PERFORMANCE METRICS"


print("\n\tACCURACY SCORE\n\t******************************\n")
print (f"\t{metrics.accuracy_score(y, y_preds)*100}")


print("\n\tCLASSICATION REPORT\n\t******************************\n")
print (f"\t{metrics.classification_report(y, y_preds)}")


"CONFUSION MATRIX"

print("\n\tPLOTTED CONFUSION MATRIX\n\t******************************\n")
shk = metrics.confusion_matrix(y, y_preds)
fig, ax = plt.subplots(figsize=(8,6))
ax= plt.subplot()
sns.heatmap(shk, annot=True, ax = ax,fmt='g'); #annot=True to annotate cells
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(["Attack","Normal"]); ax.yaxis.set_ticklabels(["Attack","Normal"]);
plt.show()
