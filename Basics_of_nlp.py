import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

#1. Download all necessary NLTK data

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

#2. Sample data
document=[
    "Data science is amaing and fun.",
    "The data scientist works with data everyday",
    "Here are the basics of nlp"
]

#3. Define Preprocessing Function
stop_words=set(stopwords.words("english"))
lemmatizer=WordNetLemmatizer()

def clean_text(text):
    tokens=nltk.word_tokenize(text.lower())
    clean_tokens=[]
    for word in tokens:
        if word.isalpha() and word not in stop_words:
            root_word = lemmatizer.lemmatize(word)
            clean_tokens.append(root_word)

    return " ".join(clean_tokens)

cleaned_docs = [clean_text(doc) for doc in document]
print("Cleaned Docs:", cleaned_docs)

#4. vectorize 

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(cleaned_docs)

print("\nFeature Names (Vocabulary):", tfidf.get_feature_names_out())
print("\nTF-IDF Matrix Shape:", X.shape)
print("\nFirst Document Vector:\n", X[0].toarray())
print("\nThe whole Document vector:\n",X.toarray())

