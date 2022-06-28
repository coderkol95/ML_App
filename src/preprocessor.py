import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

PROBLEM_TYPE = pd.read_json("options/build_options.json")["problem_types"]["default"]
COL_FREQUENCY_CONTINUOUS = 10

class preprocessor():

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

    def fit(
        self,
        numeric_imputer,
        categorical_imputer,
        numeric_scaler,
        categorical_encoder):

        self.numeric_pipeline = Pipeline(
        [
            ('imputer_num', numeric_imputer),
            ('scaler', numeric_scaler)
        ]
        )

        self.categorical_pipeline = Pipeline(
        [
            ('imputer_cat', categorical_imputer),
            ('onehot', categorical_encoder)
        ]
        )
        self.preprocessor_pipe = ColumnTransformer(
        [
            ('categoricals', self.categorical_pipeline, self.__cat_cols),
            ('numericals', self.numeric_pipeline, self.__num_cols)
        ],
        )
        self.preprocessor_pipe.fit(self.X_train)

    def fit_transform(
        self,
        X,
        numeric_imputer = None,
        categorical_imputer = None,
        numeric_scaler = None,
        categorical_encoder = None):

        if numeric_imputer == None or categorical_encoder == None or numeric_scaler == None or categorical_encoder == None:

            if hasattr(self, 'preprocessor_pipe'):
                return self.preprocessor_pipe.transform(X)

            else:
                raise ValueError("Please fit the pipeline first or pass the necessary values.")
        
        else:
            
            self.fit(numeric_imputer, categorical_imputer, numeric_scaler, categorical_encoder)
            return self.preprocessor_pipe.transform(X)

    def transform(self, X):

        if hasattr(self, 'preprocessor_pipe'):
            return self.preprocessor_pipe.transform(X)
        
        else:
            raise ValueError("Please fit the pipeline first.")
        



