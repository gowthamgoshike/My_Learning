import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Create a Dummy Dataset
# In a real project, you would load this from a CSV
data = {
    'text': [
        "Win a free iPhone now!",
        "Hey, are we still meeting for lunch?",
        "URGENT: Your account password has expired.",
        "Can you send me the project files?",
        "Exclusive offer: 50% off on all shoes",
        "Mom called, she wants you to call back.",
        "Congratulations! You've won the lottery.",
        "Let's grab coffee tomorrow morning."
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}
df = pd.DataFrame(data)

# 2. Split Data (Training vs Testing)
# We keep some data aside to test if the model actually learned
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.25, random_state=42
)

# 3. Build the Full Stack Pipeline
# This single object will Vectorize (TF-IDF) then Classify (Naive Bayes)
model = make_pipeline(
    TfidfVectorizer(stop_words='english'), 
    MultinomialNB()
)

# 4. Train the Model
print("Training model...")
model.fit(X_train, y_train)

# 5. Test the Model
# Notice we pass raw text directly! The pipeline handles the vectorization.
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)*100:.1f}%")

# 6. Live Prediction (The "User Interface" part)
new_emails = [
    "Free cash prize waiting for you",  # Should be spam
    "Hey, did you finish the report?"   # Should be ham
]
results = model.predict(new_emails)

print("\n--- Live Test Results ---")
for email, result in zip(new_emails, results):
    print(f"Email: '{email}' -> Prediction: {result.upper()}")

joblib.dump(model, 'nlp-spam-detector/models/spam_model.pkl')

print("Model saved successfully!")