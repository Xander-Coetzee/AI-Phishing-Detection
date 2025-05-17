import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)

def clean_text(text):
    """Clean and preprocess text."""
    try:
        text = str(text).strip().lower()
        if not text:
            return ""
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return str(text).strip()

def preprocess_text(text):
    """Tokenize and lemmatize text."""
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(tokens)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the data
df = pd.read_csv("phishing-email-dataset/phishing_email.csv")

# Handle missing values
df["text_combined"] = df["text_combined"].fillna("")

# Clean and preprocess text
df["text_cleaned"] = df["text_combined"].apply(clean_text)
df["text_processed"] = df["text_cleaned"].apply(preprocess_text)

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=1000,  # Limit to top 1000 features for efficiency
    ngram_range=(1, 2),  # Include unigrams and bigrams
    stop_words="english",
    min_df=5,  # Ignore terms that appear in less than 5 documents
    max_df=0.7,  # Ignore terms that appear in more than 70% of documents
)

# Fit and transform the processed text
tfidf_features = tfidf.fit_transform(df["text_processed"])

# Convert to DataFrame for easier manipulation
tfidf_df = pd.DataFrame(
    tfidf_features.toarray(),
    columns=[f"tfidf_{feature}" for feature in tfidf.get_feature_names_out()],
)

# Combine with original dataframe
df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

# Define features and target
X = df.drop(columns=["label", "text_combined", "text_cleaned", "text_processed"])
y = df["label"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Train Random Forest
print("\n=== Training Random Forest ===")
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import classification_report, accuracy_score

y_pred = rf.predict(X_test)
print("\n=== Model Evaluation ===")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(rf, 'phishing_detection_rf_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("\nModel and vectorizer saved successfully!")
print("Files created:")
print("- phishing_detection_rf_model.pkl")
print("- tfidf_vectorizer.pkl")