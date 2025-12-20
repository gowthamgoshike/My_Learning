import pandas as pd
from src.feature_engineering import ChurnFeatureEngineer

def test_feature_creation():
    df = pd.DataFrame({
        'tenure': [5, 20],
        'MonthlyCharges': [70, 90],
        'TotalCharges': [300, 1800]
    })

    fe = ChurnFeatureEngineer()
    out = fe.transform(df)

    assert 'avg_monthly_spend' in out.columns
    assert 'tenure_bucket' in out.columns
    assert 'high_monthly_charge' in out.columns
