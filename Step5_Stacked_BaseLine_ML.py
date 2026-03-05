#!/usr/bin/env python
# coding: utf-8

# # Stacked Data Baseline ML Tests
#
# Required Data File `./FullStacked_data.csv`
#
#
# ## Basic Data Preparation

import os
import sys

import numpy as np
import pandas as pd

# Dataset location
DATASET = "FullStacked_data.csv"
assert os.path.exists(DATASET)

# Load and shuffle
dataset = pd.read_csv(DATASET).sample(frac=1).reset_index(drop=True)


# #### Note: Becaues we are using `sample(frac = 1)` we are randomizing all the data. Therefore, results will vary from time to time based on the data set reading.

dataset.head()


# Drop first 3 columns and isBurnt label
# 0 index of columns - so ",3" drops  {0,1,2}
X = np.array(dataset.iloc[:, 3:-1])
y = np.array(dataset.isBurnt)
y = y - 1  # shift from {1.2} to {0,1} for non-burn, burn


# ---
#
# ## Test Base Line ML Classifiers

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# ## Baseline a resubstitution Logistic Regression

# Create an instance of a model that can be trained
model = LogisticRegression()

# fit = "train model parameters using this data and expected outcomes"
model.fit(X, y)
LR_RESUB_SCORE = model.score(X, y)
print("Logistic Regression: {0:6.5f}".format(LR_RESUB_SCORE))


# ---
#
# ## Baseline a resubstitution KNeighborsClassifier

# Create an instance of a model that can be trained
model = KNeighborsClassifier()

# fit = "train model parameters using this data and expected outcomes"
model.fit(X, y)
KNN_RESUB_SCORE = model.score(X, y)
print("KNN : {0:6.5f}".format(KNN_RESUB_SCORE))


# ---
#
# ## Baseline a resubstitution Decision Tree

# Create an instance of a model that can be trained
model = DecisionTreeClassifier()

# fit = "train model parameters using this data and expected outcomes"
model.fit(X, y)
DT_RESUB_SCORE = model.score(X, y)
print("Decision Tree: {0:6.5f}".format(DT_RESUB_SCORE))


# ---
#
# ## Baseline a resubstitution LinearSVC

# Create an instance of a model that can be trained
model = LinearSVC()

# fit = "train model parameters using this data and expected outcomes"
model.fit(X, y)
SVC_RESUB_SCORE = model.score(X, y)
print("Linear SVC Regression: {0:6.5f}".format(SVC_RESUB_SCORE))


# ---
# ## Resubstitution Model Summary
#
# * Logistic Regression: 0.88639
# * K(5) Nearest Neighbors: 0.93313
# * Decision Tree: 0.99982
# * Linear SVC: 0.80398
#
# ---

#
# ## Cross-Fold Analysis of Classifier Generalizability
# We are going to do a 5-fold cross validation for each model.
# Then, compare the degrade.

import sklearn.model_selection

XFOLD = 5


# Hide the pesky warnings from Logit
import warnings

warnings.simplefilter("ignore")

# new model
model = LogisticRegression()
# Show Prior
print("Resub Logistic Regression: {0:6.5f}".format(LR_RESUB_SCORE))
# Run Cross Val
cv_results = sklearn.model_selection.cross_val_score(model, X, y, cv=XFOLD)

for i, acc in enumerate(cv_results):
    change = (acc - LR_RESUB_SCORE) / LR_RESUB_SCORE * 100
    print("Fold {}: {:6.5f}, change {:5.2f}%".format(i, acc, change))

print("Average Logit Acc {:5.2f}%".format(np.mean(cv_results) * 100))


# new model
model = KNeighborsClassifier()
# Show Prior
print("Resub KNN: {0:6.5f}".format(KNN_RESUB_SCORE))
# Run Cross Val
cv_results = sklearn.model_selection.cross_val_score(model, X, y, cv=XFOLD)

for i, acc in enumerate(cv_results):
    change = (acc - KNN_RESUB_SCORE) / KNN_RESUB_SCORE * 100
    print("Fold {}: {:6.5f}, change {:5.2f}%".format(i, acc, change))

print("Average KNN Acc {:5.2f}%".format(np.mean(cv_results) * 100))


# new model
model = DecisionTreeClassifier()
# Show Prior
print("Resub Decision Tree: {0:6.5f}".format(DT_RESUB_SCORE))
# Run Cross Val
cv_results = sklearn.model_selection.cross_val_score(model, X, y, cv=XFOLD)

for i, acc in enumerate(cv_results):
    change = (acc - DT_RESUB_SCORE) / DT_RESUB_SCORE * 100
    print("Fold {}: {:6.5f}, change {:5.2f}%".format(i, acc, change))

print("Average Decision Tree Acc {:5.2f}%".format(np.mean(cv_results) * 100))


# new model
model = LinearSVC()
# Show Prior
print("Resub SVC: {0:6.5f}".format(SVC_RESUB_SCORE))
# Run Cross Val
cv_results = sklearn.model_selection.cross_val_score(model, X, y, cv=XFOLD)

for i, acc in enumerate(cv_results):
    change = (acc - SVC_RESUB_SCORE) / SVC_RESUB_SCORE * 100
    print("Fold {}: {:6.5f}, change {:5.2f}%".format(i, acc, change))

print("Average Linear SVC Acc {:5.2f}%".format(np.mean(cv_results) * 100))


# ## Notes
#  * Average Logit Acc 88.64%
#  * Average KNN Acc 90.67%
#  * Average Decision Tree Acc 87.67%
#  * Average Linear SVC Acc 78.55%
#
# ### The high-performing decision tree seems overfit .
#
# ### The linear Support Vector Machine is very inconsistent
#
# ### The best is the KNN with an average Accuracy of 90.67%

# ---
