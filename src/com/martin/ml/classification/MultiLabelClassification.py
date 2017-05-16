from sklearn.metrics.classification import f1_score
from sklearn.model_selection._validation import cross_val_predict
from sklearn.neighbors.classification import KNeighborsClassifier

from com.martin.ml.classification.DisplayImage import y_train, some_digit
import matplotlib.pyplot as plt
import numpy as np

from com.martin.ml.classification.Train import X_train

y_train_large=(y_train >=7)
y_train_odd = (y_train%2==1)
y_multilabel= np.c_[y_train_large,y_train_odd]
knn_clf=KNeighborsClassifier()
knn_clf.fit(X_train,y_multilabel)
print("KNN CLF",knn_clf.predict([some_digit]))
y_train_knn_pred=cross_val_predict(knn_clf,X_train,y_train,cv=3)
print("F1 score ",f1_score(y_train,y_train_knn_pred,average="macro"))