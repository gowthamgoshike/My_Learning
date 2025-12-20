from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.feature_engineering import ChurnFeatureEngineer
from src.preprocessing import preprocessor

model = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)

pipeline = Pipeline([
    ('feature_engineering', ChurnFeatureEngineer()),
    ('preprocessing', preprocessor),
    ('model', model)
])
