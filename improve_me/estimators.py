from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class Estimator1:
    _regularization: float = 1e-6
    _params: Union[np.ndarray, None] = None
    
    
    def fit(self, X, y) -> None:
        C = X.T @ X
        if np.linalg.matrix_rank(C) < C.shape[0]:
            C += self._regularization * np.eye(C.shape[0])
        self._params = np.linalg.pinv(C) @ X.T @ y
        

    def predict(self, X) -> np.ndarray:
        return X @ self._params

    @property
    def params(self) -> Union[np.ndarray, None]:
        return self._params


@dataclass
class Estimator2:
    _regularization: float = 1e-6
    _params: np.ndarray = None
    
    
    def fit(self, X, y):
        C = X.T @ X
        if np.linalg.matrix_rank(C) < C.shape[0]:
            C += self._regularization * np.eye(C.shape[0])
        self._params = np.linalg.pinv(C) @ X.T @ y
        

    def predict(self, X):
        return X @ self._params

    @property
    def params(self):
        return self._params
