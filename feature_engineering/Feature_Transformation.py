import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Sample Data: Salary is skewed (one very high value), Age is normal
df = pd.DataFrame({
    'Salary': [50000, 60000, 55000, 52000, 1000000], # Right Skewed
    'Age': [25, 30, 35, 40, 22]
})

# 1. Log Transformation (Fixing Skewness)
# We use log1p (log(1+x)) to avoid errors if value is 0
df['Salary_Log'] = np.log1p(df['Salary'])

# 2. Standardization (Scaling)
# Good because it won't be as affected by the huge salary outlier as MinMax
scaler = StandardScaler()
df['Age_Scaled'] = scaler.fit_transform(df[['Age']])

print(df)