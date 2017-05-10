from sklearn.model_selection import StratifiedKFold
from com.martin.ml.classification.DisplayImage import *
from sklearn.model_selection import cross_val_score
from com.martin.ml.classification.Train import *
from sklearn.base import clone
skfolds=StratifiedKFold(n_splits=3,random_state=42)
for train_index, test_index in skfolds.split(X_train,y_train_5):
    clone_clf=clone(sgd_clf)
    X_train_folds=X_train[train_index]
    y_train_folds=(y_train_5[train_index])
    X_test_fold=X_train[test_index]
    y_test_fold=(y_train_5[test_index])

    clone_clf.fit(X_train_folds,y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    print(y_pred)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct/len(y_pred))
    print(cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring="accuracy"))