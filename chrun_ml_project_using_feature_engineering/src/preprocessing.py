from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
NUM_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_spend']
CAT_COLS = ['Contract', 'PaymentMethod', 'tenure_bucket']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # fill NaNs with median
        ('scaler', StandardScaler())
    ]), NUM_COLS),
    ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS)
])
