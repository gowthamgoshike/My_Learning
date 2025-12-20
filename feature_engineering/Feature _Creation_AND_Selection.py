
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample Data: House Sales
df = pd.DataFrame({
    'Price': [200000, 400000, 300000, 600000],
    'Area_SqFt': [1000, 2000, 1500, 3000],
    'Date_Sold': ['2023-01-01', '2023-01-15', '2023-02-10', '2023-03-05'],
    'Temp_C': [25, 26, 24, 25], # Useless noise for house price?
})

# --- 1. Feature Creation ---

# Create 'Price_Per_SqFt'
df['Price_Per_SqFt'] = df['Price'] / df['Area_SqFt']

# Extract Month from Date
df['Date_Sold'] = pd.to_datetime(df['Date_Sold'])
df['Month_Sold'] = df['Date_Sold'].dt.month

# --- 2. Feature Selection ---

# Check Correlation
# We want to see what correlates with 'Price'
correlation = df.corr(numeric_only=True)

print("Correlation with Price:\n", correlation['Price'])

# If Temp_C has low correlation, we drop it
df = df.drop(columns=['Temp_C'])

print("\nFinal Data:\n", df)