import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Phishing Email Detector", page_icon="📧", layout="wide")

import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from urllib.parse import urlparse
import html
import unicodedata

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Phishing keywords for counting
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

# Load the trained model and TF-IDF vectorizer
try:
    model = joblib.load("phishing_detection_rf_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()


def clean_text(text):
    """Clean and preprocess text."""
    try:
        text = str(text).strip().lower()
        if not text:
            return ""
        text = html.unescape(text)
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


def extract_email_features(text):
    """Extract features from email text."""
    features = {}

    # Basic text features
    features["text_length"] = len(text)
    words = text.split()
    features["avg_word_length"] = (
        sum(len(word) for word in words) / len(words) if words else 0
    )
    features["num_uppercase"] = sum(1 for c in text if c.isupper())
    features["uppercase_ratio"] = features["num_uppercase"] / len(text) if text else 0

    # Phishing keyword count
    features["phishing_keyword_count"] = sum(
        1 for word in phishing_keywords if word in text.lower()
    )

    # Urgency indicators
    urgent_phrases = [
        "urgent",
        "immediate",
        "action required",
        "verify now",
        "account suspended",
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


def analyze_links(text):
    """Analyze links in the text."""
    links = re.findall(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        text,
    )
    num_links = len(links)
    url_mismatch = 0

    for link in links:
        try:
            domain = urlparse(link).netloc
            if re.search(r"[<>\"]", link):
                url_mismatch += 1
        except:
            continue

    return num_links, url_mismatch


def predict_email(email_text):
    """Predict if an email is phishing or legitimate."""
    # Clean and preprocess text
    cleaned_text = clean_text(email_text)
    processed_text = preprocess_text(cleaned_text)

    # Extract features
    features = extract_email_features(cleaned_text)
    num_links, url_mismatch = analyze_links(cleaned_text)
    features.update(
        {
            "num_links": num_links,
            "url_mismatch": url_mismatch,
            "has_attachments": int(
                "attach" in cleaned_text.lower() or "enclosure" in cleaned_text.lower()
            ),
        }
    )

    # Create TF-IDF features
    tfidf_features = tfidf.transform([processed_text])

    # Combine all features
    feature_df = pd.DataFrame([features])
    tfidf_df = pd.DataFrame(
        tfidf_features.toarray(), columns=tfidf.get_feature_names_out()
    )

    # Ensure all expected features are present
    all_features = pd.concat([feature_df, tfidf_df], axis=1)

    # Make prediction
    prediction = model.predict(all_features)[0]
    proba = model.predict_proba(all_features)[0][
        1
    ]  # Probability of being phishing (class 1)

    return prediction, proba, features


# Streamlit UI
st.title("📧 Phishing Email Detector")
st.write(
    "Upload an email (text file) or paste the email content below to check if it's a phishing attempt."
)

# File uploader
uploaded_file = st.file_uploader("Upload an email (TXT file)", type=["txt"])

# Text area for direct input
email_text = st.text_area("Or paste the email content here:", height=200)

if uploaded_file is not None:
    email_text = uploaded_file.read().decode("utf-8")
    st.text_area("Email content:", email_text, height=200, disabled=True)

if st.button("Analyze Email") and email_text:
    with st.spinner("Analyzing email..."):
        try:
            prediction, confidence, features = predict_email(email_text)

            # Display results
            st.subheader("🔍 Analysis Results")

            if prediction == 1:
                st.error(
                    f"🚨 **Phishing Detected!** (Confidence: {confidence*100:.1f}%)"
                )
            else:
                st.success(
                    f"✅ **Legitimate Email** (Confidence: {(1-confidence)*100:.1f}%)"
                )

            # Show key features
            st.subheader("📊 Key Features")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Text Length", features["text_length"])
                st.metric("Phishing Keywords", features["phishing_keyword_count"])
                st.metric("Urgency Score", features["urgency_score"])

            with col2:
                st.metric("Links Found", features.get("num_links", 0))
                st.metric("Contains Phone", "Yes" if features["has_phone"] else "No")
                st.metric("Contains IP", "Yes" if features["has_ip"] else "No")

            # Show explanation
            st.subheader("ℹ️ Analysis")
            if prediction == 1:
                st.warning(
                    "This email shows several characteristics commonly found in phishing attempts. Please be cautious and verify the sender's identity before taking any action."
                )
            else:
                st.info(
                    "This email appears to be legitimate. However, always remain vigilant and verify unexpected requests."
                )

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")

# Add some styling
st.markdown(
    """
    <style>
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Add footer
st.markdown("---")
st.caption("Phishing Email Detector v1.0 | Built with Streamlit")
