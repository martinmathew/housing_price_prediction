from com.martin.ml.classification.ConfusionMatrix import y_train_pred
from com.martin.ml.classification.DisplayImage import X_train, y_train
import matplotlib.pyplot as plt
import numpy as np
cl_a,cl_b=3,5
X_aa=X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab=X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba=X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb=X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
plt.figure(figsize=(8,8))
plt.subplot(221)
plot_digits(X_aa[:25],images_per_row=5)