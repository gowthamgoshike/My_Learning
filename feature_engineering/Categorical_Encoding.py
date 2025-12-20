import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Sample Data
df = pd.DataFrame({
    'Size': ['S', 'L', 'M', 'XL', 'S'],      # Order matters
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue'] # No order
})

# 1. Ordinal Encoding (For Size)
# We must specify the order manually so it knows S < M < L < XL
rank_order = [['S', 'M', 'L', 'XL']]
encoder = OrdinalEncoder(categories=rank_order)

df['Size_Encoded'] = encoder.fit_transform(df[['Size']])
# Result: S=0, M=1, L=2, XL=3

# 2. One-Hot Encoding (For Color)
# Creates new columns: Color_Blue, Color_Green, Color_Red
df = pd.get_dummies(df, columns=['Color'], prefix='Color')

print(df)