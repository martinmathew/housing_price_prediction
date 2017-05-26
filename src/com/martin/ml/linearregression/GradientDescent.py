import numpy as np

from com.martin.ml.linearregression.NormalEquation import X_b

n_iteration=1000
eta=0.1
m=100
theta=np.random.randn(2,1)
print("Theta : ",theta)
for iteration in range(n_iteration):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta))
    theta = theta - eta*gradients
print("Final Theta",theta)