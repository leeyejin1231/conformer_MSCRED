import pandas as pd
from typing import Union, List
import numpy as np


def MinMaxScaler(data: Union[pd.DataFrame, List], max_=None, min_=None):
    if isinstance(data, pd.DataFrame):
        X = data.to_numpy()
    else:
        X = np.array(data)

    if not isinstance(max_, np.ndarray):
        max_ = X.max(axis=0)
        min_ = X.min(axis=0)
    X_std = (X - min_) / (max_ - min_)
    # X_scaled = X_std * (max_ - min_) + min_
    X_scaled = X_std

    return X_scaled, max_, min_
