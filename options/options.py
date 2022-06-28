from re import M
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler,OneHotEncoder, OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll import scope


scaling_params = {
    "MinMaxScaler":{
                    "feature_range":(0,1),
                    "clip":[False, True][0]
                    },
    "StandardScaler":{
                    "with_mean":[True, False][0],
                    "with_std":[True, False][0]
                    }
}

encoding_params = {
    "OneHotEncoder":{
                    "drop":["first", None,"if_binary"][0],
                    "handle_unknown": ["ignore", "infrequent_if_exist","error"][0],
                    "min_frequency":0.2,
                    "max_categories":None
                    },
    "OrdinalEncoder":{
                    "handle_unknown" : ["use_encoded_value","error"][0],
                    "unknown_value": 99
                    }
}

imputation_params = {
    "SimpleImputer":{
                    "missing_values":[np.nan, "?"][0],
                    "strategy":["mean", "median", "most_frequent", "constant"][1],
                    "fill_value": None
                    },
    "IterativeImputer":{
                        "missing_values" : [np.nan, "?"][0],
                        "max_iter" : 10,
                        "tol" : 1e-3,
                        "n_nearest_features" : None,
                        "initial_strategy" : ["mean", "median", "most_frequent", "constant"][0],
                        "imputation_order" : ["ascending", "descending", "random"][0],
                        "random_state":42
                        },
    "KNNImputer":{
                    "missing_values" : [np.nan, "?"][0],
                    "n_neighbors" : 5,
                    "weights" : ["uniform", "distance"][0],
                    "metric" : "nan_euclidean"
                }
}

preprocessing_opts = {   
    "scaling":{
                "MinMaxScaler":MinMaxScaler().set_params(**scaling_params['MinMaxScaler']), 
                "StandardScaler":StandardScaler().set_params(**scaling_params['StandardScaler'])
                },
    "encoding":{
                "OneHotEncoder":OneHotEncoder().set_params(**encoding_params['OneHotEncoder']), 
                "OrdinalEncoder":OrdinalEncoder().set_params(**encoding_params['OrdinalEncoder'])
                },
    "imputation":{
                "SimpleImputer":SimpleImputer().set_params(**imputation_params['SimpleImputer']),
                "IterativeImputer":IterativeImputer().set_params(**imputation_params['IterativeImputer']), 
                "KNNImputer":KNNImputer().set_params(**imputation_params['KNNImputer'])
                }
}

ml_algos = {
    "LinearRegression" : LinearRegression(),
    "LogisticRegression" : LogisticRegression(),
    "KNeighborsClassifier" : KNeighborsClassifier(),
    "KNeighborsRegressor" : KNeighborsRegressor(),
    "GaussianNB" : GaussianNB(),
    "DecisionTreeClassifier" : DecisionTreeClassifier(),
    "DecisionTreeRegressor" : DecisionTreeRegressor()
}

param_spaces ={
"GradientBoostingRegressor" :  
                            {
                                'learning_rate': hp.uniform('learning_rate',0.01,1),
                                'n_estimators': scope.int(hp.quniform('n_estimators',10,500,1))
                            },
"LinearRegression" : 
                    {
                    'fit_intercept':hp.choice('fit_intercept',[True, False]),
                    'positive':hp.choice('positive',[True, False])
                    }
}