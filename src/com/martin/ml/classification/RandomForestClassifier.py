from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.ranking import roc_curve, roc_auc_score
from sklearn.model_selection._validation import cross_val_predict
import matplotlib.pyplot as plt

from com.martin.ml.classification.Threshold import fpr, tpr, plot_roc_curve
from com.martin.ml.classification.Train import X_train, y_train_5

forest_clf=RandomForestClassifier(random_state=42)
y_probas_forest=cross_val_predict(forest_clf,X_train,y_train_5,cv=3,method="predict_proba")
y_scores_forest=y_probas_forest[:,1]
fpr_forest,tpr_forest,thresholds_forest=roc_curve(y_train_5,y_scores_forest)
plt.plot(fpr,tpr,"b:",label="SGD")
plot_roc_curve(fpr_forest,tpr_forest,"Random Forest")
plt.legend(loc="bottom right")
print("ROC AUC SCORE ",roc_auc_score(y_train_5,y_scores_forest))