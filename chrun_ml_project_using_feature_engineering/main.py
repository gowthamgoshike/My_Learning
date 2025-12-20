import pandas as pd
from sklearn.model_selection import train_test_split
from src.pipeline import pipeline
from src.evaluation import evaluate_model


# Load dataset (example)
df = pd.read_csv('data/raw/rawdata.csv')


X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})


X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)


pipeline.fit(X_train, y_train)
evaluate_model(pipeline, X_test, y_test)