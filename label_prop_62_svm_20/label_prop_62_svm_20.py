#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 22:44:58 2020

@author: hteza
"""

# as I can recall from the class
# We have to model SVM with 20 labelled data
# but the label propagation with the full dataset ( except for specified 30% 50% and 100%)

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.semi_supervised import LabelSpreading

df_colon=pd.read_csv("/Users/hteza/Desktop/Class/RADS602/colon.csv")

#######################################################################

# for propagation
X = df_colon [{'T62947', 'H64807'}]
X = X.to_numpy()
y = df_colon['Class']
y = np.where(y==1,y,0)

rng = np.random.RandomState(0)


y_30 = np.copy(y)
y_30[rng.rand(len(y))<0.3]=-1

y_50 = np.copy(y)
y_50[rng.rand(len(y))<0.5]=-1

# e here is just a blank variable
e = 0
ls30 = (LabelSpreading().fit(X,y_30),y_30,e)
ls50 = (LabelSpreading().fit(X,y_50),y_50,e)
ls100 = (LabelSpreading().fit(X,y),y,e)

#######################################################################

# for SVM
df_sam = df_colon [{'T62947', 'H64807', 'Class'}]

# randomly select 42 samples without replacement
# they will become unlabelled
df_unlab = df_sam.sample(n=42, replace=False, random_state=1)
#df_unlab.to_csv("df_unlabelled_42_before_unlabelling.csv")

# to find out the rest, merge the 42 with original in a new dataframe
# those rows existing on original only means they are not in 42 samples
# they will be labelled

df_lab = pd.merge(df_sam, df_unlab, how='outer', indicator=True)
df_lab = df_lab.loc[df_lab._merge=='left_only',['T62947', 'H64807', 'Class']]
#df_lab.to_csv("df_labelled_20.csv")

X_train = df_lab [{'T62947', 'H64807'}]
X_train = X_train.to_numpy()
y_train = df_lab ['Class']
y_train = np.where(y_train==1,y_train,0)

#X_test = df_unlab [{'T62947', 'H64807'}]
#X_test = X_test.to_numpy()
#y_test = df_unlab ['Class']
#y_test = np.where(y_test==1,y_test,0)

# svm parameters
rbf_svc_lin = (svm.SVC(kernel='linear',gamma=0.5).fit(X_train,y_train),y_train, 'linear')
rbf_svc_poly = (svm.SVC(kernel='poly',gamma=0.5).fit(X_train,y_train),y_train, 'poly')
rbf_svc_rbf = (svm.SVC(kernel='rbf',gamma=0.5).fit(X_train,y_train),y_train, 'gaussian')

#######################################################################

# mesh
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), 
                     np.arange(y_min, y_max, h))

# titles
titles = ['Label Spreading 30% data', 
          'Label Spreading 50% data', 
          'Label Spreading 100% data']

# -1 for unlabelled, 0 for negative, 1 for positive
color_map = {-1: (1, 1, 1), 
             0: (0, 0, 0.9), 
             1: (1, 0, 0)}

for i, (clf, y_train, svm_type) in enumerate((ls30, ls50, ls100, rbf_svc_lin, rbf_svc_poly, rbf_svc_rbf)):
    plt.subplot(3, 2, i + 1)
    if i<3:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        contour_opts = {'levels': 2,
                        'cmap': plt.cm.Pastel1}
        plt.contourf(xx, yy, Z, **contour_opts) 
        plt.contour(xx, yy, Z, **contour_opts) 
        plt.axis('off')
        colors = [color_map[y] for y in y_train]
        plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black')
        plt.title(titles[i])
    if i>=3:
        kernel = svm_type
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        contour_opts = {'levels': 2,
                        'cmap': plt.cm.Pastel1}
        plt.contourf(xx, yy, Z, **contour_opts) 
        plt.contour(xx, yy, Z, **contour_opts) 
        plt.axis('off')
        colors = [color_map[y] for y in y_train]
        plt.scatter(X_train[:, 0], X_train[:, 1], c=colors, edgecolors='black')
        plt.title('SVM ('+ kernel+')')
    i=i+1
        
plt.suptitle("Unlabeled points are colored white", y=0.1)
plt.show()
