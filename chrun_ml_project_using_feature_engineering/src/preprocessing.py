from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


NUM_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_spend']
CAT_COLS = ['Contract', 'PaymentMethod', 'tenure_bucket']


preprocessor = ColumnTransformer([
('num', StandardScaler(), NUM_COLS),
('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS)
])