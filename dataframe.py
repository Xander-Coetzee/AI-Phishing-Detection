import pandas as pd
import re
import nltk
import numpy as np
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import email.utils
from urllib.parse import urlparse
import html
from typing import Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Load the data
df = pd.read_csv("phishing-email-dataset/phishing_email.csv")

# Handle missing values
df["text_combined"] = df["text_combined"].fillna("")


def clean_text(text):
    """
    Clean and preprocess text with enhanced patterns for phishing detection.
    Handles HTML, dates, times, emails, URLs, and other common patterns.

    Args:
        text (str): Input text to clean

    Returns:
        str: Cleaned and normalized text
    """
    try:
        # Convert to string and normalize whitespace
        text = str(text).strip().lower()
        if not text:
            return ""

        # Decode HTML entities and remove HTML tags
        text = html.unescape(text)
        text = re.sub(r"<[^>]+>", " ", text)

        # Pre-compile patterns for better performance
        patterns = [
            # Dates
            (
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
                " <DATE> ",
            ),  # MM/DD/YYYY or DD-MM-YY
            (r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", " <DATE> "),  # YYYY-MM-DD
            (
                r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s-]\d{1,2}(?:st|nd|rd|th)?,?[\s-]\d{2,4}\b",
                " <DATE> ",
            ),
            (
                r"\b\d{1,2}[\s-](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s-]\d{2,4}\b",
                " <DATE> ",
            ),
            # Times
            (r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[ap]m|AM|PM)?\b", " <TIME> "),
            # Phone numbers
            (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", " <PHONE> "),
            (r"\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b", " <PHONE> "),
            (
                r"\b\+?\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b",
                " <PHONE> ",
            ),
            # Email addresses (simple pattern)
            (r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", " <EMAIL> "),
            # URLs (http, https, ftp)
            (r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w .-]*", " <URL> "),
            # IP addresses
            (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", " <IP> "),
        ]

        # Apply all patterns
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Handle file extensions (separate to avoid false positives in URLs)
        text = re.sub(
            r"\.(exe|zip|rar|7z|js|vbs|bat|cmd|ps1|sh)\b",
            " <EXECUTABLE> ",
            text,
            flags=re.IGNORECASE,
        )

        # Handle currency and amounts
        text = re.sub(r"\$\d+(?:,\d{3})*(?:\.\d{2})?", " <CURRENCY> ", text)

        # Replace long digit sequences (IDs, account numbers, etc.)
        text = re.sub(r"\b\d{5,}\b", " <LONG_NUM> ", text)

        # Normalize unicode characters
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )

        # Remove remaining special characters but keep words and basic tokens
        text = re.sub(r"[^\w\s.<>]", " ", text)

        # Normalize whitespace and return
        return re.sub(r"\s+", " ", text).strip()

    except Exception as e:
        print(f"Error cleaning text: {e}")
        return str(text).strip()  # Return original text if error occurs

    return text


# Apply text cleaning
df["text_cleaned"] = df["text_combined"].apply(clean_text)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words("english"))
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(tokens)


# Apply preprocessing
df["text_processed"] = df["text_cleaned"].apply(preprocess_text)

# Display data info
print("DataFrame Info:")
print(df.info())
print("\nLabel distribution:")
print(df["label"].value_counts())

# Show sample of cleaned and processed text with more details
print("\n=== Text Preprocessing Examples ===")
print("Original text:")
print("-" * 80)
print(df["text_combined"].iloc[0][:150] + "...")
print("\nCleaned text:")
print("-" * 80)
print(df["text_cleaned"].iloc[0][:150] + "...")
print("\nProcessed text (after tokenization, stopword removal, etc.):")
print("-" * 80)
print(df["text_processed"].iloc[0][:150] + "...")

# Show more examples of processed text
print("\n=== More Processed Text Examples ===")
for i in range(1, 5):
    print(f"\nExample {i}:")
    print("-" * 40)
    print(f"Original: {df['text_combined'].iloc[i][:100]}...")
    print(f"Processed: {df['text_processed'].iloc[i][:100]}...")

# Extract metadata features
print("\nExtracting metadata features...")


# 1. Sender domain analysis
def extract_sender_domain(email_str):
    try:
        parsed = email.utils.parseaddr(email_str)
        if parsed[1]:
            domain = parsed[1].split("@")[-1]
            return domain.lower()
    except:
        pass
    return ""


df["sender_domain"] = df["text_combined"].apply(extract_sender_domain)

# 2. Subject keyword flags
phishing_keywords = [
    "urgent",
    "account",
    "password",
    "verify",
    "security",
    "update",
    "confirm",
    "login",
    "alert",
    "warning",
]


def count_phishing_keywords(text):
    return sum(1 for word in phishing_keywords if word in text.lower())


df["phishing_keyword_count"] = df["text_combined"].apply(count_phishing_keywords)


# 3. Link analysis
def analyze_links(text):
    # Count total links
    links = re.findall(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        text,
    )
    num_links = len(links)

    # Check for URL-text mismatch
    url_mismatch = 0
    for link in links:
        try:
            # Extract domain from URL
            domain = urlparse(link).netloc
            # Check if URL is hidden in HTML
            if re.search(r'[<>"]', link):
                url_mismatch += 1
        except:
            continue

    return num_links, url_mismatch


# Apply link analysis
df[["num_links", "url_mismatch"]] = (
    df["text_combined"].apply(analyze_links).apply(pd.Series)
)


# 4. Enhanced Email Structure Features
def extract_email_features(text):
    """Extract structural features from email text."""
    features = {}

    # HTML features
    features["has_html"] = int(bool(re.search(r"<[a-z][\s\S]*>", text)))
    features["num_images"] = len(re.findall(r"<img[^>]+>", text, re.IGNORECASE))
    features["has_forms"] = int(
        bool(re.search(r"<form|<input|<button", text, re.IGNORECASE))
    )
    features["has_attachments"] = int(
        bool(re.search(r"attach|enclosure|see attached", text, re.IGNORECASE))
    )

    # Text statistics
    features["text_length"] = len(text)
    features["avg_word_length"] = (
        np.mean([len(word) for word in text.split()]) if text else 0
    )
    features["num_uppercase"] = sum(1 for c in text if c.isupper())
    features["uppercase_ratio"] = features["num_uppercase"] / len(text) if text else 0

    # Punctuation and special characters
    features["exclamation_count"] = text.count("!")
    features["question_count"] = text.count("?")
    features["dollar_count"] = text.count("$")
    features["percent_count"] = text.count("%")

    # Urgency indicators
    urgent_phrases = [
        "urgent",
        "immediate",
        "action required",
        "verify now",
        "account suspended",
        "security alert",
        "limited time",
        "act now",
        "click below",
        "confirm your account",
    ]
    features["urgency_score"] = sum(
        text.lower().count(phrase) for phrase in urgent_phrases
    )

    # Suspicious patterns
    features["has_phone"] = int(bool(re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", text)))
    features["has_ip"] = int(
        bool(re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text))
    )

    return features


# Apply enhanced feature extraction
print("\nExtracting enhanced features...")
email_features = df["text_combined"].apply(extract_email_features).apply(pd.Series)
df = pd.concat([df, email_features], axis=1)

# Add domain-based features
df["is_free_domain"] = (
    df["sender_domain"]
    .str.contains(
        r"gmail|yahoo|hotmail|outlook|aol|protonmail|zoho|yandex|mail\.ru|inbox\.com",
        case=False,
        regex=True,
        na=False,
    )
    .astype(int)
)

# Add time-based features (if you have timestamp data)
# df['hour_sent'] = pd.to_datetime(df['timestamp']).dt.hour
# df['is_work_hours'] = df['hour_sent'].between(9, 17).astype(int)

# Display new features
print("\nNew features added:")
feature_columns = [
    "sender_domain",
    "phishing_keyword_count",
    "num_links",
    "url_mismatch",
    "has_html",
    "num_images",
    "has_forms",
    "has_attachments",
    "text_length",
    "avg_word_length",
    "uppercase_ratio",
    "exclamation_count",
    "question_count",
    "dollar_count",
    "percent_count",
    "urgency_score",
    "has_phone",
    "has_ip",
    "is_free_domain",
]
print(df[feature_columns].head().to_string())

# Display feature correlations with label (if available)
if "label" in df.columns:
    print("\nFeature correlations with label:")
    # Select only numeric columns for correlation
    numeric_cols = (
        df[feature_columns + ["label"]]
        .select_dtypes(include=[np.number])
        .columns.tolist()
    )
    if numeric_cols:
        correlations = df[numeric_cols].corr()["label"].sort_values(ascending=False)
        print(correlations.to_string())
    else:
        print("No numeric columns available for correlation analysis")

# Display statistics of numeric features
print("\nStatistics of numeric features:")
# Get all numeric columns except label
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != "label"]

if numeric_cols:
    print(df[numeric_cols].describe().to_string())
else:
    print("No numeric features available for statistics")

# Extract TF-IDF features
print("\nExtracting TF-IDF features...")

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

print(f"\nAdded {tfidf_features.shape[1]} TF-IDF features")

# Show top 20 most important TF-IDF features
print("\n=== Top 20 Most Important TF-IDF Features ===")
tfidf_sums = tfidf_df.sum().sort_values(ascending=False)
print(tfidf_sums.head(20))

# Show sample of TF-IDF features for first few emails
print("\n=== Sample TF-IDF Features for First 5 Emails ===")
print(tfidf_df.iloc[:5, :10])

# Prepare data for modeling
print("\n=== Preparing Data for Modeling ===")

# Select features (all numeric columns except the label)
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
# Remove 'label' from features if it exists
if "label" in numeric_cols:
    numeric_cols.remove("label")

X = df[numeric_cols]
y = df["label"]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Train Random Forest with hyperparameter tuning
print("\n=== Training Random Forest with Hyperparameter Tuning ===")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Define the parameter grid for hyperparameter tuning
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": ["balanced", "balanced_subsample"],
}

# Create base model
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# Grid search with cross-validation
grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1
)

print("Starting grid search (this may take a few minutes)...")
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_
print("\nBest parameters found:")
print(grid_search.best_params_)

# Make predictions
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\n=== Model Evaluation ===")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame(
    {"feature": X_train.columns, "importance": best_rf.feature_importances_}
).sort_values("importance", ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))

# Save the trained model
import joblib

joblib.dump({"model": best_rf, "vectorizer": tfidf}, "phishing_detector_bundle.pkl")
print("\nModel saved as 'phishing_detection_rf_model.pkl'")
