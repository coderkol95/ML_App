import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer

from src.preprocessing import COL_FREQUENCY_CONTINUOUS

PROBLEM_TYPE = pd.read_json("options/build_options.json")["problem_types"]["default"]
COL_FREQUENCY_CONTINUOUS = 10

class preprocessor:

    def __init__(self, X, y):

        self.X = pd.DataFrame(X)
        self.y = pd.DataFrame(y)
        self.__num_cols = list(self.X.columns[[i>COL_FREQUENCY_CONTINUOUS for i in self.X.nunique().tolist()]])
        self.__cat_cols = list(set(self.X.columns).difference(set(self.__num_cols)))

    def split(
        self,
        test_pct: float = 0.2,
        problem_type: str = PROBLEM_TYPE):

        if problem_type == "regression":
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_pct, shuffle=True, random_state=42)

        elif problem_type == "classification":
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_pct, shuffle=True, random_state=42, stratify = y)

    def pipeline(
        self,
        # numeric_imputer,
        # categorical_imputer,
        # numeric_scaler,
        # categorical_encoder
        ):

        self.split()

        numeric_pipeline = Pipeline(
        [
            ('imputer_num', KNNImputer()),
            ('scaler', StandardScaler())
        ]
        )

        categorical_pipeline = Pipeline(
        [
            ('imputer_cat', KNNImputer()),
            ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
        ]
        )
        self.preprocessor = ColumnTransformer(
        [
            ('categoricals', categorical_pipeline, self.__cat_cols),
            ('numericals', numeric_pipeline, self.__num_cols)
        ],
        )
        
        return self.preprocessor

