from orca.scripts import self_voicing
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics.classification import confusion_matrix
from sklearn.model_selection._validation import cross_val_score, cross_val_predict
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from com.martin.ml.classification.DisplayImage import some_digit, X_train as xtr
from com.martin.ml.classification.RandomForestClassifier import forest_clf
from com.martin.ml.classification.Train import sgd_clf, y_train, X_train

sgd_clf.fit(X_train, y_train)
print("Multi Class", sgd_clf.predict([some_digit]))
some_digit_scores = sgd_clf.decision_function([some_digit])
print("Decision Score", some_digit_scores)
print(sgd_clf.classes_)
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
print("OVO", ovo_clf.predict([some_digit]))
forest_clf.fit(X_train, y_train)
print("Forrest Multi classifier", forest_clf.predict_proba([some_digit]))
print("Cross val score ", cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
print("Cross val score scaling : ", cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix", conf_mx)
y_train_pred = cross_val_predict(sgd_clf,X_train_scaled,y_train,cv=3)
conf_mx=confusion_matrix(y_train,y_train_pred)
plt.matshow(conf_mx,cmap=plt.cm.gray)
row_sums=conf_mx.sum(axis=1,keepdims=True)
norm_conf_mx=conf_mx/row_sums
np.fill_diagonal(norm_conf_mx,0)
plt.matshow(norm_conf_mx,cmap=plt.cm.gray)
plt.show()
