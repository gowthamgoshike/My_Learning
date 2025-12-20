import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Sample Data with missing values
data = pd.DataFrame({
    'Age': [25, 30, np.nan, 22, 100],  # 100 is an outlier
    'City': ['Hyd', 'Bglr', np.nan, 'Hyd', 'Chennai']
})

# 1. Median Imputation (numeric)
num_imputer = SimpleImputer(strategy='median')
data['Age'] = num_imputer.fit_transform(data[['Age']]).ravel()

# 2. Mode Imputation (categorical)
cat_imputer = SimpleImputer(strategy='most_frequent')
data['City'] = cat_imputer.fit_transform(data[['City']]).ravel()

print(data)
