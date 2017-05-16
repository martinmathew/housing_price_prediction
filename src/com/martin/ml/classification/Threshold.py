from louis import plain_text
from sklearn.metrics.classification import precision_score, recall_score
from sklearn.metrics.ranking import precision_recall_curve, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection._validation import cross_val_predict

from com.martin.ml.classification.DisplayImage import some_digit
from com.martin.ml.classification.Train import sgd_clf, X_train, y_train_5

def plot_precision_recall_vs_threshold(precision,recalls,thresholds):
    plt.plot(thresholds,precision[:-1],"b--",label="Precision")
    plt.plot(thresholds,recalls[:-1],"g-",label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])


def plot_roc_curve(fpr,tpr,label=None):
    plt.plot(fpr,tpr,linewidth=2,label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

y_scores=sgd_clf.decision_function([some_digit])
print("Y Scores ")
for p in y_scores:
    print(p)

threshold=0
y_some_digit_pred = (y_scores > threshold)
threshold = 0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
y_scores=cross_val_predict(sgd_clf,X_train,y_train_5,cv=3,method="decision_function")
precisions,recalls,thresholds = precision_recall_curve(y_train_5,y_scores)

y_train_pred_90 = (y_scores > 70000)
print("Precision score " ,precision_score(y_train_5,y_train_pred_90))
print("Recall score ", recall_score(y_train_5,y_train_pred_90))
fpr,tpr,threshold=roc_curve(y_train_5,y_scores)
plot_roc_curve(fpr,tpr)
print("ROC AUC Score",roc_auc_score(y_train_5,y_scores))
