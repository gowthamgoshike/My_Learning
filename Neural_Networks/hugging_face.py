from transformers import pipeline

# 1. Initialize the pipeline
# This downloads a default pre-trained model for sentiment analysis
classifier = pipeline("sentiment-analysis")

# 2. Define the text you want to analyze
text_data = [
    "I love learning new technologies, it makes me feel powerful!",
    "I am frustrated when my code crashes and I can't find the bug."
]

# 3. Run the model
print("Analyzing text...")
results = classifier(text_data)

# 4. Print results
for text, result in zip(text_data, results):
    print(f"Text: '{text}'")
    print(f"Label: {result['label']}, Score: {round(result['score'], 4)}\n")