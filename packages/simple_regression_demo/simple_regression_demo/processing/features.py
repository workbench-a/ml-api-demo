import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DemoTransformer(BaseEstimator, TransformerMixin):
    """Demo transformer."""
    
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # to accomodate the pipeline
        return self
    
    def transform(self, X):
        X = X.copy()
        
#         # handle errors
        
        # return all but first data point in the array
        for feature in self.variables:
            X[feature] = 0.5*X[feature]
        return X