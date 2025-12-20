import pandas as pd
from src.pipeline import pipeline

def test_pipeline_fit_predict():
    X = pd.DataFrame({
        'tenure': [5, 20],
        'MonthlyCharges': [70, 90],
        'TotalCharges': [300, 1800],
        'Contract': ['Month-to-month', 'Two year'],
        'PaymentMethod': ['Electronic check', 'Credit card']
    })

    y = [1, 0]

    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == 2
