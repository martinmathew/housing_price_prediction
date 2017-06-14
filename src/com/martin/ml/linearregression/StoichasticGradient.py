import numpy as np
from sklearn.linear_model.stochastic_gradient import SGDRegressor

from com.martin.ml.linearregression.GradientDescent import m
from com.martin.ml.linearregression.NormalEquation import X_b, Y, X

n_epochs=50
t0,t1=5,50

def learning_schedule(t):
    return t0/(t+t1)

theta = np.random.randn(2,1)
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = Y[random_index:random_index+1]
        gradient = 2 * xi.T.dot(xi.dot(theta)-yi)
        eta = learning_schedule(epoch *m+1)
        theta = theta - eta * gradient
print("Stoichastic Theta : ",theta)
sgd_reg = SGDRegressor(n_iter=n_epochs,penalty=None,eta0 = 0.1,)
sgd_reg.fit(X,Y.ravel())
print("Stoic api : ",sgd_reg.intercept_)