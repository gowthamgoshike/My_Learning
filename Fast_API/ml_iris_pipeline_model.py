
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Get Data
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2. Rename columns to be simple for your API
X.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# 3. Create a Pipeline (Scaler + Model)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=10))
])

# 4. Train the "Brain"
pipeline.fit(X, y)

# 5. Save it as a Pickle file
joblib.dump(pipeline, "final_pipeline_v1.pkl")

print("✅ Success! 'final_pipeline_v1.pkl' has been created in your folder.")