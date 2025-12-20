from sklearn.linear_model import LogisticRegression


model = LogisticRegression(
class_weight='balanced',
max_iter=500,
random_state=42
)