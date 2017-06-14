from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_name):
        self.attribute_name =  attribute_name
    def fit(self,X,y=None):
        return self

    def transform(self,X):
        return X[self.attribute_name].values