from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model.logistic import LogisticRegression

iris = datasets.load_iris()
print(list(iris.keys()))
X=iris["data"][:,3:]
Y=(iris["target"] == 2).astype(np.int)
log_reg = LogisticRegression()
log_reg.fit(X,Y)
X_new = np.linspace(0,3,1000).reshape(-1,1)
print("X_new : ",X_new)
Y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new,Y_proba[:,1],"g-",label="Iris-Virginica")
plt.plot(X_new,Y_proba[:,0],"g-",label="Not - Virginica")
plt.show()


