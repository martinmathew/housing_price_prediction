import numpy.random as rnd

from com.martin.ml.classification.DisplayImage import X_train, X_test
from com.martin.ml.classification.MultiLabelClassification import knn_clf
import matplotlib.pyplot as plt
noise=rnd.randint(0,100,(len(X_train),784))
noise=rnd.randint(0,100,(len(X_test),784))
X_train_mod=X_train+noise
X_test_mode=X_test+noise
y_train_mod=X_train
y_test_mode=X_test
knn_clf.fit(X_train_mod,y_train_mod)
clean_digit=knn_clf.predict([X_test_mode[99]])
plot_digit
