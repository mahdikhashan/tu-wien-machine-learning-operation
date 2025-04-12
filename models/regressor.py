import xgboost as xgb


class XGBQuantileError():
    def __init__(
            self, 
            objective="reg:quantileerror", 
            quantile_alpha=0.25,
            learning_rate=0.01,
            max_depth=5,
            tree_method="hist",
            eval_metric='rmse'
        ):
        self.objective = objective
        self.quantile_alpha = quantile_alpha
        self.eval_metric = eval_metric
        self.lr = learning_rate
        self.max_depth = max_depth
        self.tree_method = tree_method
        self.model = None

    @property
    def params(self):
        return {
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'max_depth': self.max_depth,
            'learning_rate': self.lr,
            'tree_method': self.tree_method,
            'quantile_alpha': self.quantile_alpha
        }

    def train(self, X_train, y_train, X_val, y_val, config: dict):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        watchlist = [(dtrain, 'train'), (dval, 'eval')]        
        self.model = xgb.train(
            self.params, dtrain, 
            num_boost_round=config["num_boost_round"], 
            evals=watchlist, 
            early_stopping_rounds=config["early_stopping_rounds"]
        )

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def evaluate(self, X_val, y_val, evaluator):
        y_pred = self.predict(X_val)
        eval_score = evaluator(y_val, y_pred)
        return eval_score


class LowerBoundModel(XGBQuantileError):
    def __init__(self):
        super().__init__(quantile_alpha=0.1)


class HigherBoundModel(XGBQuantileError):
    def __init__(self):
        super().__init__(quantile_alpha=0.9)
