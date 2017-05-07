from com.martin.ml.classification.FetchData import *
import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]
digit=y[36000]
print(digit)
some_digit_image=some_digit.reshape(28,28)
plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()