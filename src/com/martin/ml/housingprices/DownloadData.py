import hashlib

import os
import tarfile
import pandas as pd
from matplotlib.pyplot import plot
from pandas.plotting._misc import scatter_matrix
from six.moves import urllib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.model_selection._split import StratifiedShuffleSplit
from sklearn.preprocessing.data import OneHotEncoder
from sklearn.preprocessing.imputation import Imputer
from sklearn.preprocessing.label import LabelEncoder, LabelBinarizer

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH="datasets/housing"
HOUSING_URL=DOWNLOAD_ROOT+HOUSING_PATH+"/housing.tgz"

def split_train_test(data,test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

def fetch_housing_data(housing_url=HOUSING_URL,housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def test_set_check(identifier,test_ratio,hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data,test_ratio,id_column,hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_,test_ratio,hash))
    return data.loc[~in_test_set],data.loc[in_test_set]

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)
#fetch_housing_data()
housing = load_housing_data()
#train_set,test_set = split_train_test(housing,0.2)
housing_with_id = housing.reset_index()
housing_with_id["index"] = housing["longitude"]*1000+housing["latitude"]
train_set,test_set = train_test_split(housing,test_size=2,random_state=42)
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0 , inplace=True)

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_test = housing.loc[train_index]
    strat_test_test = housing.loc[test_index]


for set in (strat_test_test,strat_train_test):
    set.drop(["income_cat"],axis=1,inplace=True)

#housing = strat_train_test.copy()


corr_matrix = housing.corr()

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
housing = strat_train_test.drop("median_house_value",axis = 1)
housing_labels = strat_train_test["median_house_value"].copy

imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)
X=imputer.transform(housing_num)
housing_tr = pd.DataFrame(X,columns=housing_num.columns)

housing_cat = housing["ocean_proximity"]
#print(housing.info)
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
#print(housing_cat_1hot)



