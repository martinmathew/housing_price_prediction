import numpy as np
from com.martin.ml.classification.DisplayImage import *
from sklearn.linear_model import SGDClassifier
shuffle_index=np.random.permutation(60000)
X_train,y_train = X[shuffle_index],y[shuffle_index]
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_5)
print(sgd_clf.predict([some_digit]))
print(digit)
