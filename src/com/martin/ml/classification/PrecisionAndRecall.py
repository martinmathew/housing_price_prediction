from sklearn.metrics import recall_score,precision_score

from com.martin.ml.classification.ConfusionMatrix import y_train_pred
from com.martin.ml.classification.CrossValidation import y_pred
from com.martin.ml.classification.Train import y_train_5

print(precision_score(y_train_5,y_train_pred))
print(recall_score(y_train_5,y_train_pred))