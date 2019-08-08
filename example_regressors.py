import sklearn

from matplotlib import pyplot
import numpy as np

lasso = linear_model.Lasso()

data = np.random.randn(100,2)
stability_score = np.random.randn(100)
lasso.fit(data, stability_score)
ss_predicted = lasso.predict(data)

fig,ax = pyplot.subplots(1,1)
ax.scatter(stability_score, ss_predicted)
fig.show()
ss_predicted
ss_predicted = lasso.predict(data)
ss_predicted
lasso.coef_
lasso.intercept_
stability_score = 0.1*data[:,0] - 0.4*data[:,1] + np.random.randn(100)*0.1
lasso.fit(data, stability_score)
ss_predicted = lasso.predict(data)
lasso.coef_
ax.cla()
pyplot.ion()
ax.scatter(data, c=stability_score)
ax.scatter(data[:,0], data[:,1], c=stability_score)
lasso.fit(data,stability_score)
lasso.coef_
stability_score = 0.1*data[:,0] - 0.4*data[:,1]
lasso.fit(data,stability_score)
lasso.coef_
from sklearn import svm
svmr = svm.LinearSVR()
svmr.fit(data,stability_scores)
svmr.fit(data,stability_score)
svmr.coef_
stability_score = 0.1*data[:,0] - 0.4*data[:,1] + 0.1*np.random.randn(100)
svmr.fit(data,stability_score)
svmr.coef_
ss_predicted = svmr.predict(data)
ax.cla()
ax.scatter(stability_score, ss_predicted)
ax.plot([-1,1],[-1,1], c='k')

# A collection of regressors to try.

from sklearn import linear_model
lasso = linear_model.Lasso()
from sklearn import ensemble
rf = ensemble.RandomForestRegressor()
from sklearn import svm
svr = svm.LinearSVR()
from sklearn import neighbors
knr = neighbors.KNeighborsRegressor()

