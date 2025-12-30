import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# 1. Create sample data (House Size vs Price)
data = {'size_sqft': [1000, 1500, 2000, 2500, 3000],
        'price': [200000, 300000, 400000, 500000, 600000]}
df = pd.DataFrame(data)

# 2. Train Model
X = df[['size_sqft']]
y = df['price']
model = LinearRegression()
model.fit(X, y)

# 3. Save Model
# We verify if the folder exists
if not os.path.exists('model'):
    os.makedirs('model')

joblib.dump(model, 'Docker_Learning/model/housing_model.pkl')
print("Model trained and saved to Docker_Learning/model/housing_model.pkl")