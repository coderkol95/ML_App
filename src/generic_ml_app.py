import pandas as pd
import numpy as np
from pyparsing import col
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.impute import KNNImputer, IterativeImputer
import pickle

from sklearn.datasets import make_regression

X, y = make_regression(n_features=10, n_samples=1000, n_informative=7, shuffle=True)

# TODO-preprocessing
# 1. Split into train/test



def _remove_rows(
    X,
    min_pct:float = None):

    if min_pct==None:
        raise ValueError("No value mentioned for min_pct")

    else:
        X.dropna(thresh=int(np.floor(min_pct * X.shape[1])), inplace=True)
        return X

def _KNN_impute(
    X:pd.DataFrame,
    mode:str):

    if mode == "train":
        k = KNNImputer().fit(X)
        with open('artefacts/preprocessing/KNN_Imputer.pkl', 'wb') as K:
            pickle.dump(k, K)
        X = k.transform(X)
        return X
    
    elif mode == "test":
        with open ('artefacts/preprocessing/KNN_Imputer.pkl' , 'rb') as K:
            k = pickle.load(K)
            X = k.transform(X)
            return X

def _iteratively_impute(
    X:pd.DataFrame,
    mode:str):

    if mode == "train":
        i = IterativeImputer().fit(X)
        with open('artefacts/preprocessing/Iterative_Imputer.pkl', 'wb') as I:
            pickle.dump(i, I)
        X = i.transform(X)
        return X
    
    elif mode == "test":
        with open ('artefacts/preprocessing/Iterative_Imputer.pkl', 'rb') as I:
            i = pickle.load(I)
            X = i.transform(X)
            return X

# 2. Impute missing values
def impute(
    X , #pd.DataFrame | np.ndarray,
    method: str = "KNN",
    mode: str = None,
    min_pct: float = 1) -> np.array:

    if type(X) != pd.core.frame.DataFrame:
        cols = [f"col{i+1}" for i in np.arange(X.shape[1])]
        X = pd.DataFrame(X, columns=cols)

    __cols = X.columns

    if method == "drop":
        X = _remove_rows(X=X, min_pct=min_pct)    

    elif method == "KNN":
        X = _KNN_impute(X=X, mode=mode)

    elif method == "Iterative":
        X = _iteratively_impute(X=X, mode=mode)

    return pd.DataFrame(X, columns=__cols)
    







# 3. Scale the data / Encoding
# 4. Feature selection 
