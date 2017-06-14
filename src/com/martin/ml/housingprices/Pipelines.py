from com.martin.ml.housingprices.CombinedAttributesAdder import  CombinedAttributesAdder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing.data import StandardScaler
from sklearn.preprocessing.imputation import Imputer
from sklearn.preprocessing.label import LabelBinarizer

from com.martin.ml.housingprices.DataFrameSelector import DataFrameSelector
from com.martin.ml.housingprices.DownloadData import housing_num, housing

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector',DataFrameSelector(num_attribs)),
                         ('imputer',Imputer(strategy = "median")),
                         ('attribs_adder',CombinedAttributesAdder()),
                         ('std_scaler',StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector',DataFrameSelector(cat_attribs)),
                         ('label_binarizer',LabelBinarizer()),
])

full_pipeline = FeatureUnion(transformer_list=[("num_pipeline",num_pipeline),
                                               ("cat_pipeline",cat_pipeline)])
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)