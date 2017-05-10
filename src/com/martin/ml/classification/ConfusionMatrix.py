from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from com.martin.ml.classification.CrossValidation import y_pred
from com.martin.ml.classification.Train import y_train_5, sgd_clf, X_train
from sklearn.metrics import recall_score,precision_score

y_train_pred=cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)
print(confusion_matrix(y_train_5,y_train_pred))
