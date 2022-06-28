import numpy as np
import pandas as pd
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll import scope
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

PROBLEM_TYPE = pd.read_json("options/build_options.json")["problem_types"]["default"]

class modeller:

    def __init__(self, model, scoring_technique, param_space, max_evals, problem_type = PROBLEM_TYPE):

        self.model = model
        self.scoring_technique = scoring_technique
        self.problem_type = problem_type
        self.param_space = param_space
        self.max_evals = max_evals
        if self.problem_type == "classification":
            self.splits=StratifiedKFold(n_splits=5, shuffle=True)
        elif self.problem_type == "regression":
            self.splits = KFold(n_splits=5, shuffle=True)
    
    def fit(self, X, y):

        def optimize(params):

            mod = self.model.set_params(**params)
            score=cross_val_score(mod,X,y,scoring=self.scoring_technique,cv=self.splits).mean()    
            
            if self.scoring_technique in ['f1', 'accuracy_score']:
                neg_multiplier = -1
            else:
                neg_multiplier = 1
            
            return neg_multiplier * score.mean()

        trials=Trials()

        def score_hyperparams(params):
            score=optimize(params)
            return {'loss':score, 'status':STATUS_OK}

        result = fmin(
            fn=score_hyperparams,
            max_evals=self.max_evals,
            space=self.param_space,
            trials=trials,
            algo=tpe.suggest)

        self.model.set_params(**result).fit(X,y)

    def predict(self, X):

        return self.model.predict(X)




