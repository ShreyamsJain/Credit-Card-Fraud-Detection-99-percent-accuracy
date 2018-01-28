"""
Credit Card Fraud Detection
----------
- Transform data to contain numeric values
- Resampled data by oversampling using RandomOverSampler
- Evaluated all the algorithms for accuracy
- Chose best 2 algorithms, scaled and estimated accuracy
- Achieved 100% accuracy, [ precision, recall and F1 score = 1.0]
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

# Loading the dataset
# Removing header names as they are not meaningful
url = 'creditcard.csv'
dataset = read_csv(url, header = None)

#Shape
print(dataset.shape)

#types
set_option('display.max_rows', 500)
print(dataset.dtypes)

print(type(dataset))

dataset = dataset.apply(pd.to_numeric, errors = 'coerce')

print(dataset.dtypes)

# head
set_option('precision', 3)
print(dataset.describe())

print(dataset.groupby(30).size())

dataset.hist(sharex = False, sharey = False, xlabelsize = 1, ylabelsize = 1)
pyplot.show()

array = dataset.values
inf_indices = np.where(np.isinf(array))
nan_indices = np.where(np.isnan(array))
print(inf_indices, type(inf_indices))
print(nan_indices, type(nan_indices))
for row, col in zip(*inf_indices):
    array[row,col] = -1
    
for row, col in zip(*nan_indices):
    array[row,col] = 0
#array[]
X = array[:, 0:30]
y = array[:, 30]
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_sample(X, y)

validation_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res,
    test_size=validation_size, random_state=42)

# Test options and evaluation metric
num_folds = 10
scoring = 'accuracy'

models = []
models.append(('LR' , LogisticRegression()))
models.append(('LDA' , LinearDiscriminantAnalysis()))
models.append(('KNN' , KNeighborsClassifier()))
models.append(('CART' , DecisionTreeClassifier()))
models.append(('NB' , GaussianNB()))
#models.append(('SVM' , SVC()))

results = []
names = []
for name, model in models:
  kfold = KFold(n_splits=num_folds, random_state=42)
  cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',
    LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA',
    LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
    KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
    DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',
    GaussianNB())])))
#pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))
results = []
names = []
for name, model in pipelines:
  kfold = KFold(n_splits=num_folds, random_state=42)
  cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model_knn = KNeighborsClassifier()
model_knn.fit(rescaledX, y_train)
# estimate accuracy on validation dataset
rescaledValidationX = scaler.transform(X_test)
predictions = model_knn.predict(rescaledValidationX)
print "accuracy score:"
print(accuracy_score(y_test, predictions))
print "confusion matrix: "
print(confusion_matrix(y_test, predictions))
print "classification report: "
print(classification_report(y_test, predictions))

model_dt = DecisionTreeClassifier()
model_dt.fit(rescaledX, y_train)
rescaledValidationX = scaler.transform(X_test)
predictions = model_dt.predict(rescaledValidationX)
print "accuracy score:"
print(accuracy_score(y_test, predictions))
print "confusion matrix: "
print(confusion_matrix(y_test, predictions))
print "classification report: "
print(classification_report(y_test, predictions))
