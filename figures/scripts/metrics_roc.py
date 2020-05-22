# -*- coding: utf-8 -*-
"""
Created on Sat May  9 11:03:34 2020

@author: Abdeljalil
"""

import numpy as np
np.random.seed(10)

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline

n_estimator = 10
X, y = make_classification(n_samples=8000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# It is important to train the ensemble of trees on a different subset
# of the training data than the linear regression model to avoid
# overfitting, in particular if the total number of leaves is
# similar to the number of training samples
X_train, X_train_lr, y_train, y_train_lr = train_test_split(
    X_train, y_train, test_size=0.5)

# Unsupervised transformation based on totally random trees
rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
                          random_state=0)

rt_lm = LogisticRegression(max_iter=1000)
pipeline = make_pipeline(rt, rt_lm)
pipeline.fit(X_train, y_train)
y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)

# Supervised transformation based on random forests
rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
rf_enc = OneHotEncoder()
rf_lm = LogisticRegression(max_iter=1000)
rf.fit(X_train, y_train)
rf_enc.fit(rf.apply(X_train))
rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)

# Supervised transformation based on gradient boosted trees
grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc = OneHotEncoder()
grd_lm = LogisticRegression(max_iter=1000)
grd.fit(X_train, y_train)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

y_pred_grd_lm = grd_lm.predict_proba(
    grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)

# The gradient boosted model by itself
y_pred_grd = grd.predict_proba(X_test)[:, 1]
fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)

# The random forest model by itself
y_pred_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
#%%

fig, axis = plt.subplots(1,1, figsize=(4.5,3.5))

axis.plot([0, 1], [0, 1], 'k--', lw='.8')
auc_rf = roc_auc_score(y_test, y_pred_rf_lm)
auc_1 = roc_auc_score(y_test, y_pred_rt)
auc_grd = roc_auc_score(y_test, y_pred_grd_lm)

c1 = mlp.cm.tab20c(0)
c2 = mlp.cm.tab20c(4)
c3 = mlp.cm.tab20c(8)
axis.plot(fpr_grd_lm, tpr_grd_lm,c=c3, label='Classifier 3 (AUC={:04.3f})'.format(auc_grd)) #gbt
axis.plot(fpr_rf, tpr_rf, c=c2,label='Classifier 2 (AUC={:04.3f})'.format(auc_rf)) #rf
axis.plot(fpr_rt_lm, tpr_rt_lm, c=c1,label='Classifier 1 (AUC={:04.3f})'.format(auc_1))





axis.set_xlabel('False Positive Rate')
axis.set_ylabel('True Positive Rate')
axis.grid(ls=':')
axis.axis([0, 1, 0, 1])
axis.legend(framealpha=.5, loc='lower right')
fig.tight_layout()

#axis.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
#axis.plot(fpr_grd, tpr_grd, label='GBT')
fig.savefig('D:\\Thesis\\Document\\figures\\metrics_roc.pdf')