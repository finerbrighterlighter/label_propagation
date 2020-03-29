#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:56:55 2020

@author: hteza
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.semi_supervised import LabelSpreading

df_colon=pd.read_csv("/Users/hteza/Desktop/Class/RADS602/colon.csv")

# label '-1' for negative will be replaced to be '0'
# because label propagation consider -1 as unlabelled

#######################################################################

# label propagation will only use 20 labelled and 42 unlabelled
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
#X_train = X_train.to_numpy()
y_train = df_lab ['Class']
y_train = np.where(y_train==1,y_train,0)

# unlabelling 42
df_unlab["Class"].replace({-1:-1 , 1: -1}, inplace=True)
#df_unlab.to_csv("df_unlabelled_42_after_unlabelling.csv")

# concating the rest of unlablled to have a full 62 again
X_train_unlab = df_unlab [{'T62947', 'H64807'}]
y_train_unlab = df_unlab ['Class']

# concating for full dataset
X_100 = pd.concat([X_train,X_train_unlab], ignore_index=True)
X_100 = X_100.to_numpy()


rng = np.random.RandomState(0)

# since I want to have 30% of 20 labelled samples
# I concat only after unlabelling the lablled
y_30 = np.copy(y_train)
y_30[rng.rand(len(y_train))<0.3]=-1
y_30 = np.append(y_30,y_train_unlab)


y_50 = np.copy(y_train)
y_50[rng.rand(len(y_train))<0.5]=-1
y_50 = np.append(y_50,y_train_unlab)


y_100 = np.append(y_train,y_train_unlab)

#######################################################################

# SVM will use full data
X = df_colon [{'T62947', 'H64807'}]
X = X.to_numpy()
y = df_colon['Class']
y = np.where(y==1,y,0)

#######################################################################

ls30 = (LabelSpreading().fit(X_100,y_30),y_30)
ls50 = (LabelSpreading().fit(X_100,y_50),y_50)
ls100 = (LabelSpreading().fit(X_100,y_100),y_100)
rbf_svc = (svm.SVC(kernel='rbf',gamma=0.5).fit(X,y),y)

#######################################################################

# mesh

h=0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), 
                     np.arange(y_min, y_max, h))

#######################################################################

titles = ['Label Spreading 30% data (6)', 
          'Label Spreading 50% data (10)', 
          'Label Spreading 100% data (20)', 
          'SVC with rbf kernel (62)']

# -1 for unlabelled, 0 for negative, 1 for positive
color_map = {-1: (1, 1, 1), 
             0: (0, 0, 0.9), 
             1: (1, 0, 0)}

for i, (clf, y_train) in enumerate((ls30, ls50, ls100, rbf_svc)):
    plt.subplot(2, 2, i + 1)
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
    
plt.suptitle("Unlabeled points are colored white", y=0.1)
plt.show()

