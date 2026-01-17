import streamlit as st
import joblib

# 1. Load the trained model
# This loads the Vectorizer AND the Naive Bayes classifier in one go
model = joblib.load('models/spam_model.pkl')

# 2. Build the UI
st.title("📧 Spam vs. Ham Detector")
st.write("Enter a message below to check if it looks like Spam.")

# 3. Get User Input
user_input = st.text_area("Message Content:", height=150)

# 4. Make Prediction
if st.button("Analyze"):
    if user_input:
        # The pipeline handles tokenization and vectorization automatically!
        prediction = model.predict([user_input])[0]
        
        # Display Result
        if prediction == 'spam':
            st.error(f"🚨 This message looks like **SPAM**.")
        else:
            st.success(f"✅ This message looks like **HAM** (Safe).")
            
        # Optional: Show the probability confidence
        proba = model.predict_proba([user_input])
        st.info(f"Confidence: {proba.max() * 100:.2f}%")
    else:
        st.warning("Please enter some text first.")