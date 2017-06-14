import numpy as np
from orca.punctuation_settings import degree
from sklearn.linear_model.base import LinearRegression
from sklearn.metrics.regression import mean_squared_error
from sklearn.model_selection._split import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing.data import PolynomialFeatures
import matplotlib.pyplot as plt
m=100
X=6*np.random.rand(m,1)-3
Y=0.5 * X**2 + X + 2+np.random.randn(m,1)
#poly_features = PolynomialFeatures(degree=2,include_bias=False)
#X_poly = poly_features.fit_transform(X)
#lin_reg = LinearRegression()
#lin_reg.fit(X_poly,Y)

def plot_learning_curves(model,X,Y):
    X_train,X_val,Y_train,Y_val = train_test_split(X,Y,test_size=0.2)
    train_errors,val_errors =[], []
    for m in range(1,len(X_train)):
        model.fit(X_train[:m],Y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, Y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, Y_val))
    plt.plot(np.sqrt(train_errors),"r-+",linewidth=2,label="train")
    plt.plot(np.sqrt(val_errors),"b-",linewidth=3,label="val")
    plt.show()
#lin_reg = LinearRegression()
#plot_learning_curves(lin_reg, X, Y)
polynomial_regression = Pipeline((("poly_feature",PolynomialFeatures(degree=10,include_bias=False)),("sgd_reg",LinearRegression())))
plot_learning_curves(polynomial_regression,X,Y)