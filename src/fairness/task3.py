import xgboost as xgb
import numpy as np
import pandas as pd

class FairDecision:
    def __init__(self, d_fcb, delta_fcb, data, delta, po_diff_sign,
                 po_transform, xgb_mod, xgb_params, x0, x1, model,
                 X, W, Z, Y, D, method, tune_params, nboot1, nboot2):
        self.d_fcb = d_fcb
        self.delta_fcb = delta_fcb
        self.data = data
        self.delta = delta
        self.po_diff_sign = po_diff_sign
        self.po_transform = po_transform
        self.xgb_mod = xgb_mod
        self.xgb_params = xgb_params
        self.x0 = x0
        self.x1 = x1
        self.model = model
        self.X = X
        self.W = W
        self.Z = Z
        self.Y = Y
        self.D = D
        self.method = method
        self.tune_params = tune_params
        self.nboot1 = nboot1
        self.nboot2 = nboot2

def fair_decisions(data, X, Z, W, Y, D, x0, x1,
                   xgb_params=None, xgb_nrounds=100,
                   po_transform=(lambda x: x), po_diff_sign=0,
                   method='medDML', model='ranger',
                   tune_params=False, nboot1=1, nboot2=100, **kwargs):

    assert isinstance(data, pd.DataFrame), "Input `data` must be a pandas DataFrame"
    xgb_params = xgb_params or {"eta": 0.1}
    method = method.lower()
    model = model.lower()

    # Fairness cookbook for original decision variable D
    d_fcb = fairness_cookbook(
        data, X=X, Z=Z, W=W, Y=D, x0=x0, x1=x1,
        model=model, method=method,
        tune_params=tune_params, nboot1=nboot1, nboot2=nboot2, **kwargs
    )

    # Fit XGBoost model to predict Y from X, Z, W, D
    feature_cols = X + Z + W + [D]
    dmatrix = xgb.DMatrix(data[feature_cols], label=data[Y])

    xgbcv = xgb.cv(
        params=xgb_params,
        dtrain=dmatrix,
        num_boost_round=xgb_nrounds,
        nfold=10,
        early_stopping_rounds=5,
        verbose_eval=False
    )

    best_nrounds = len(xgbcv)
    xgb_mod = xgb.train(
        params=xgb_params,
        dtrain=dmatrix,
        num_boost_round=best_nrounds,
        verbose_eval=False
    )

    # Predict potential outcomes and compute delta
    delta = predict_delta(xgb_mod, data, X, Z, W, D, po_diff_sign, po_transform)
    data = data.copy()
    data["delta"] = delta

    # Fairness cookbook for benefit delta
    delta_fcb = fairness_cookbook(
        data, X=X, Z=Z, W=W, Y="delta", x0=x0, x1=x1,
        model=model, method=method,
        tune_params=tune_params, nboot1=nboot1, nboot2=nboot2, **kwargs
    )

    return FairDecision(
        d_fcb=d_fcb,
        delta_fcb=delta_fcb,
        data=data,
        delta=delta,
        po_diff_sign=po_diff_sign,
        po_transform=po_transform,
        xgb_mod=xgb_mod,
        xgb_params=xgb_params,
        x0=x0,
        x1=x1,
        model=model,
        X=X,
        W=W,
        Z=Z,
        Y=Y,
        D=D,
        method=method,
        tune_params=tune_params,
        nboot1=nboot1,
        nboot2=nboot2
    )


def predict_delta(xgb_mod, data, X, Z, W, D, po_diff_sign, po_transform):
    # Create counterfactual copies of the data with D = 0 and D = 1
    data0 = data.copy()
    data1 = data.copy()
    data0[D] = 0
    data1[D] = 1

    features = X + Z + W + [D]

    dmatrix0 = xgb.DMatrix(data0[features])
    dmatrix1 = xgb.DMatrix(data1[features])

    yd0 = xgb_mod.predict(dmatrix0)
    yd1 = xgb_mod.predict(dmatrix1)

    # Enforce monotonicity constraint if needed
    if po_diff_sign == 1:
        yd1 = np.maximum(yd1, yd0)
    elif po_diff_sign == -1:
        yd1 = np.minimum(yd1, yd0)

    return po_transform(yd1) - po_transform(yd0)