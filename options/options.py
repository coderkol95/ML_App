from sklearn.preprocessing import MinMaxScaler, StandardScaler,OneHotEncoder, OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
import numpy as np

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
                    "strategy":["mean", "median", "most_frequent", "constant"][0],
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
                "MinMaxScaler":MinMaxScaler(), 
                "StandardScaler":StandardScaler()
                },
    "encoding":{
                "OneHotEncoder":OneHotEncoder(), 
                "OrdinalEncoder":OrdinalEncoder()
                },
    "imputation":{
                "SimpleImputer":SimpleImputer(),
                "IterativeImputer":IterativeImputer(), 
                "KNNImputer":KNNImputer()
                }
}