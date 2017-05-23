import numpy as np
from sklearn.linear_model import LinearRegression
X=2*np.random.rand(100,1)
Y= 4 + 3*X + np.random.rand(100,1)
X_b = np.c_[np.ones((100,1)),X]
#print(X_b)
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
print(theta_best)
X_new=np.array([[0],[2]])
X_new_b=np.c_[np.ones((2,1)),X_new]
y_predict = X_new_b.dot(theta_best)
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
print("Intercept",lin_reg.intercept_)
print("Predict: ",lin_reg.predict(X_new))