from sklearn.preprocessing import MinMaxScaler, StandardScaler,OneHotEncoder, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer

preprocessing_opts = {   
    "scaling":{
                "min_max_scaler":MinMaxScaler(), 
                "standard_scaler":StandardScaler()
                },
    "encoding":{
                "onehot_encoder":OneHotEncoder(), 
                "label_encoder":LabelEncoder()
                },
    "imputation":{
                "simple_imputer":SimpleImputer(),
                "iterative_imputer":IterativeImputer(), "knn_imputer":KNNImputer()
                }
}
