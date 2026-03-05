#!/usr/bin/env python
# coding: utf-8

# # Normalized Data Baseline ML Tests
#
# Required Data File `./normalized_global_bands_mean_stdev.csv.csv`
#
#
# ## Basic Data Preparation

import os
import sys

import numpy as np
import pandas as pd

# Dataset location
DATASET = "normalized_global_bands_mean_stdev.csv"
assert os.path.exists(DATASET)

# Load and shuffle
dataset = pd.read_csv(DATASET).sample(frac=1).reset_index(drop=True)


# #### Note: Becaues we are using `sample(frac = 1)` we are randomizing all the data. Therefore, results will vary from time to time based on the data set reading.

dataset.head()


# Drop first 3 columns and isBurnt label
# 0 index of columns - so ",4" drops  {0,1,2,3}
X = np.array(dataset.iloc[:, 4:])
y = np.array(dataset.isBurnt)
y = y - 1  # shift from {1.2} to {0,1} for non-burn, burn


# ---
#
# ## Test Baseline ML Classifiers

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
# * Logistic Regression: 0.89320
# * K(5) Nearest Neighbors: 0.93273
# * Decision Tree: 0.99982
# * Linear SVC: 0.89324
#
# ### Question: How to these resubstitution scores differ from unnormalized?  (5 pts)
# ### Thoughts on why? (5 pts)
#
# ---
#
# ## Cross-Fold Analysis of Classifier Generalizability
# We are going to do a 5-fold cross validation for each model.
# Then, compare the degrade.
# How to these resubstitution scores differ from unnormalized? (5 pts)
# Compared to the unnormalized data, the resubstitution scores for Logistic Regression (0.89320 vs. 0.89292) and Linear SVC (0.89324 vs. 0.89322) remain almost identical, showing minimal improvement. However, the K-Nearest Neighbors (0.93273 vs. 0.93350) score slightly decreased, and the Decision Tree (0.99982 in both cases) remains unchanged.Thoughts on why? (5 pts)
# I think normalization standardizes the data to a consistent scale (0 to 1), which typically benefits distance-based models like KNN and SVM by improving numerical stability and convergence. However, in this case, the results suggest that the original data was already well-scaled, leading to minimal changes for Logistic Regression and SVC. The slight drop in KNN’s performance could be due to subtle shifts in feature distances after normalization. Meanwhile, Decision Trees are inherently insensitive to scaling since they split data based on feature thresholds rather than distance calculations, explaining why its score remains the same
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
#  * Average Logist Reg Acc 89.32%
#  * Average KNN Acc 90.60%
#  * Average Decision Tree Acc 86.72%
#  * Average Linear SVC Acc 89.32%
#
# ## Questions:
# ### Based on the numbers above, which method seems to be overfitting the worst? (2 pts)
#
# ### How can you tell if it is overfitting? (2 pts)
#
# ### Which is the most consistent with the normalized data? (2 pts)
#
# ### Which is the best model and what is it's accuracy? (2 pts)
# Which method seems to be overfitting the worst? (2 pts)
# The Decision Tree Classifier is overfitting the worst. It has a resubstitution accuracy of 99.98%, but its cross-validation accuracy drops significantly to 86.79%, indicating that it performs exceptionally well on the training data but poorly on unseen data.How can you tell if it is overfitting? (2 pts)
# Overfitting is evident when there is a large gap between resubstitution accuracy (training performance) and cross-validation accuracy (performance on unseen data). The Decision Tree shows a drop of over 13%, which is much larger than other models, confirming severe overfitting.Which is the most consistent with the normalized data? (2 pts)
# The Logistic Regression and Linear SVC models are the most consistent with the normalized data, both showing an average accuracy of 89.32% with minimal fluctuations across cross-validation folds.Which is the best model and what is its accuracy? (2 pts)
# The K-Nearest Neighbors (KNN) model is the best model, achieving the highest cross-validation accuracy of 90.70%, making it the most generalizable and effective model for this dataset.
# ---

# # Take best classifer, do train/test split and Confusion Matrix

# Create an instance of a model that can be trained
model = KNeighborsClassifier()

from sklearn.model_selection import train_test_split

# This function returns four sets:
# Training features
#       # Testing features
#       #        # Training labels
#       #        #        # Testing labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# fit = "train model parameters using this data and expected outcomes"
model.fit(X_train, y_train)


get_ipython().run_line_magic("matplotlib", "inline")
import itertools

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score


# Function borrowed from:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()


y_pred = model.predict(X_test)
print(y_pred)
# pred_class = np.argmax(y_pred, axis=1)


np.set_printoptions(precision=2)
cnf_matrix = confusion_matrix(y_test, y_pred)


plt.figure()
plot_confusion_matrix(
    cnf_matrix,
    classes=["Non-Burn", "Burnt"],
    normalize=True,
    title="Normalized confusion matrix",
)

plt.show()


# ## Question:
# ### Please interpret the Confusion Matrix above.  (5 pts)
# The normalized confusion matrix shows that the K-Nearest Neighbors (KNN) classifier performs well in identifying non-burnt areas, correctly classifying 97% of them, with only 3% misclassified as burnt. However, it struggles with detecting burnt areas, correctly identifying only 35%, while misclassifying 65% as non-burnt. This high false negative rate suggests that the model is biased towards predicting non-burnt areas, possibly due to class imbalance or feature limitations. To improve burnt area detection, techniques such as class balancing (e.g., SMOTE), adjusting decision thresholds, or trying alternative models like Random Forest or boosting methods could be beneficial.
