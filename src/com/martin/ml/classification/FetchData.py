from sklearn.datasets import fetch_mldata
mnist=fetch_mldata('MNIST original')
print(mnist)
X,y = mnist["data"],mnist["target"]
print(X.shape)
print(y.shape)