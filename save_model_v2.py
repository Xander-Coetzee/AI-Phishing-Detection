import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from urllib.parse import urlparse
import html
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)

def clean_text(text):
    """Enhanced text cleaning with better HTML and URL handling."""
    try:
        text = str(text).strip().lower()
        if not text:
            return ""
        
        # Decode HTML entities and remove HTML tags
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Handle URLs more gracefully
        text = re.sub(r'https?://\S+', 'URL', text)
        
        # Remove special characters while preserving words
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return str(text).strip()

def preprocess_text(text):
    """Improved preprocessing with better tokenization."""
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(tokens)

def extract_email_features(text):
    """Extract additional features to help distinguish legitimate emails."""
    features = {}
    
    # Text statistics
    words = text.split()
    features["text_length"] = len(text)
    features["word_count"] = len(words)
    features["avg_word_length"] = sum(len(word) for word in words) / len(words) if words else 0
    
    # URL analysis
    urls = re.findall(r'https?://\S+', text)
    features["num_urls"] = len(urls)
    features["has_https"] = int(any(url.startswith('https') for url in urls))
    
    # Email structure features
    features["has_greeting"] = int(bool(re.search(r'\b(dear|hi|hello|good morning|good afternoon)\b', text)))
    features["has_signature"] = int(bool(re.search(r'\b(best regards|sincerely|thank you)\b', text)))
    
    # Professional indicators
    features["has_meeting"] = int(bool(re.search(r'\b(meeting|conference|appointment)\b', text)))
    features["has_time"] = int(bool(re.search(r'\b(\d{1,2}:\d{2}\s*(am|pm)?)\b', text)))
    features["has_date"] = int(bool(re.search(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', text)))
    
    return features

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load and preprocess data
df = pd.read_csv("phishing-email-dataset/phishing_email.csv")
df["text_combined"] = df["text_combined"].fillna("")
df["text_cleaned"] = df["text_combined"].apply(clean_text)

# Extract additional features
df["text_length"] = df["text_cleaned"].apply(len)
df["word_count"] = df["text_cleaned"].apply(lambda x: len(x.split()))
df["avg_word_length"] = df["text_cleaned"].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()) if len(x.split()) > 0 else 0)
df["num_urls"] = df["text_cleaned"].apply(lambda x: len(re.findall(r'https?://\S+', x)))
df["has_https"] = df["text_cleaned"].apply(lambda x: int(any(url.startswith('https') for url in re.findall(r'https?://\S+', x))))

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=1000,  # Reduced for better stability
    ngram_range=(1, 2),  # Using bigrams
    stop_words="english",
    min_df=0.0001,  # Very low threshold
    max_df=0.99,  # Very high threshold
    sublinear_tf=True
)

# Create feature matrix
X = df["text_cleaned"]
y = df["label"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Vectorize the text
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Create and train the model
print("\n=== Training Random Forest ===")
rf = RandomForestClassifier(
    n_estimators=100,  # Standard number of trees
    max_depth=20,      # Reasonable depth
    min_samples_split=5,  # Standard split criteria
    min_samples_leaf=2,   # Standard leaf criteria
    random_state=42,
    n_jobs=-1,
    class_weight={0: 0.55, 1: 0.45}  # Slightly favor legitimate emails
)

rf.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = rf.predict(X_test_tfidf)
y_pred_proba = rf.predict_proba(X_test_tfidf)[:, 1]

print("\n=== Model Evaluation ===")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nF1 Score (weighted):", f1_score(y_test, y_pred, average='weighted'))

# Save the model and vectorizer
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(rf, 'phishing_detection_model.pkl')

print("\nModel saved successfully!")
print("Files created:")
print("- tfidf_vectorizer.pkl")
print("- phishing_detection_model.pkl")
