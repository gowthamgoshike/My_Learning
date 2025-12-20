import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['avg_monthly_spend'] = X['TotalCharges'] / X['tenure'].replace(0, 1)
        X['tenure_bucket'] = pd.cut(
            X['tenure'], bins=[0,12,24,48,72],
            labels=['0-1yr','1-2yr','2-4yr','4-6yr']
        )
        X['high_monthly_charge'] = (X['MonthlyCharges'] > X['MonthlyCharges'].median()).astype(int)
        return X
