import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
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

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


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


def evaluate_model():
    print("Loading model and vectorizer...")
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load the model and vectorizer
    model_path = os.path.join(script_dir, "phishing_detection_rf_model.pkl")
    tfidf_path = os.path.join(script_dir, "tfidf_vectorizer.pkl")

    try:
        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        print("Model and vectorizer loaded successfully!")
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
        return

    print("Loading dataset...")
    # Load the data
    df = pd.read_csv("phishing-email-dataset/phishing_email.csv")

    # Handle missing values
    df["text_combined"] = df["text_combined"].fillna("")

    # Clean and preprocess text
    print("Preprocessing data...")
    df["text_cleaned"] = df["text_combined"].apply(clean_text)
    df["text_processed"] = df["text_cleaned"].apply(preprocess_text)

    # Transform the processed text with TF-IDF
    print("Extracting features...")
    tfidf_features = tfidf.transform(df["text_processed"])

    # Convert to DataFrame for easier manipulation
    tfidf_df = pd.DataFrame(tfidf_features.toarray())
    
    # Combine with original dataframe
    df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

    # Get the feature names from the model
    model_feature_names = model.feature_names_in_
    print(f"Model expects {len(model_feature_names)} features")

    # Create a DataFrame with the expected feature names
    # First, get all TF-IDF feature names
    tfidf_feature_names = [f"tfidf_{feature}" for feature in tfidf.get_feature_names_out()]

    # Create a DataFrame with all zeros
    X = pd.DataFrame(0, index=range(len(df)), columns=model_feature_names)

    # Fill in the TF-IDF features
    for i, feature_name in enumerate(tfidf_feature_names):
        if feature_name in model_feature_names:
            X[feature_name] = tfidf_features.toarray()[:, i]

    # Add any additional features that might be in the model
    for feature in model_feature_names:
        if not feature.startswith('tfidf_'):
            if feature == 'text_length':
                X[feature] = df['text_cleaned'].apply(len)
            elif feature == 'num_uppercase':
                X[feature] = df['text_cleaned'].apply(lambda x: sum(1 for c in x if c.isupper()))
            elif feature == 'phishing_keyword_count':
                phishing_keywords = [
                    "urgent", "account", "password", "verify", "security", 
                    "update", "confirm", "login", "alert", "warning"
                ]
                X[feature] = df['text_cleaned'].apply(
                    lambda x: sum(1 for word in phishing_keywords if word in x.lower())
                )

    # Define target
    y = df["label"]

    # Split the data using the same split as in training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Print metrics
    print("\n===== Model Evaluation Metrics =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\n===== Classification Report =====")
    print(classification_report(y_test, y_pred))

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Save the confusion matrix
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as 'confusion_matrix.png'")

    # Save metrics to a text file for the report
    with open("model_metrics.txt", "w") as f:
        f.write("===== Model Evaluation Metrics =====\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n\n")
        f.write("===== Classification Report =====\n")
        f.write(classification_report(y_test, y_pred))

    print("Metrics saved to 'model_metrics.txt'")

    # Return metrics for further use if needed
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
    }


if __name__ == "__main__":
    evaluate_model()
