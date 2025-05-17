from flask import Flask, render_template_string, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from urllib.parse import urlparse
import html
import pandas as pd
import signal
import traceback
import time

import threading


class TimeoutError(Exception):
    pass


def with_timeout(timeout):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [None]
            error = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    error[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                raise TimeoutError(f"Operation timed out after {timeout} seconds")
            if error[0] is not None:
                raise error[0]
            return result[0]

        return wrapper

    return decorator


# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)

# Initialize Flask app
app = Flask(__name__)

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

# Load the model and TF-IDF vectorizer
print("Loading model and TF-IDF vectorizer...")
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model and vectorizer using absolute paths
model_path = os.path.join(script_dir, "phishing_detection_rf_model.pkl")
tfidf_path = os.path.join(script_dir, "tfidf_vectorizer.pkl")

print(f"Loading model from: {model_path}")
print(f"Loading TF-IDF vectorizer from: {tfidf_path}")

try:
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    model_loaded = True

    # Log model information
    print("\nModel Information:")
    print(f"Model type: {type(model)}")
    print(
        f"Model features: {model.feature_names_in_ if hasattr(model, 'feature_names_in_') else 'Not available'}"
    )
    print(f"Model classes: {model.classes_}")

    # Log TF-IDF information
    print("\nTF-IDF Vectorizer Information:")
    print(f"Vocabulary size: {len(tfidf.vocabulary_)}")
    print(f"First 10 features: {list(tfidf.vocabulary_.keys())[:10]}")

except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False


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


def analyze_links(text):
    """Analyze links in the text."""
    if not text:
        return 0, 0

    # Simple URL pattern that's less likely to cause regex errors
    url_pattern = r"https?://\S+"

    try:
        # First try with a simple pattern
        links = re.findall(url_pattern, text)
        num_links = len(links)
        url_mismatch = 0

        for link in links:
            try:
                # Clean up the URL by removing any trailing punctuation
                clean_link = link.rstrip(".,;:!?)\"'")
                # Parse the URL to validate it
                parsed = urlparse(clean_link)
                if not all([parsed.scheme, parsed.netloc]):
                    continue
                # Check for suspicious characters
                if re.search(r"[<>\"]", clean_link):
                    url_mismatch += 1
            except Exception as e:
                print(f"Error processing URL {link}: {e}")
                continue

        return num_links, url_mismatch

    except Exception as e:
        print(f"Error in analyze_links: {e}")
        # Return default values if there's an error
        return 0, 0


def predict_email(email_text):
    """Predict if an email is phishing or legitimate."""
    if not model_loaded:
        return {"is_phishing": False, "confidence": 0, "error": "Model not loaded"}

    try:
        if not email_text or not email_text.strip():
            return {
                "is_phishing": False,
                "confidence": 0,
                "error": "Email text is empty",
            }

        # Clean and preprocess text
        cleaned_text = clean_text(email_text)
        if not cleaned_text:
            return {
                "is_phishing": False,
                "confidence": 0,
                "error": "No valid text to analyze",
            }

        processed_text = preprocess_text(cleaned_text)

        # Create TF-IDF features first
        tfidf_features = tfidf.transform([processed_text])

        # Log TF-IDF features
        print("\nTF-IDF Features:")
        print(f"Feature names: {tfidf.get_feature_names_out()}")
        if app.debug:
            print(f"Feature values shape: {tfidf_features.shape}")

        # Get all feature names the model expects
        all_feature_names = (
            model.feature_names_in_ if hasattr(model, "feature_names_in_") else []
        )

        # Initialize all features to 0
        features = {name: 0 for name in all_feature_names}

        # Log model features
        print("\nModel Features:")
        print(f"Expected features: {all_feature_names}")

        # Set basic features if they exist in the model's expected features
        if "text_length" in features:
            features["text_length"] = len(cleaned_text)

        if "num_uppercase" in features:
            features["num_uppercase"] = sum(1 for c in cleaned_text if c.isupper())

        if "phishing_keyword_count" in features:
            features["phishing_keyword_count"] = sum(
                1 for word in phishing_keywords if word in cleaned_text.lower()
            )

        # Add personal email indicators
        if "personal_greeting" in features:
            greetings = ["dear", "hi", "hello", "hey"]
            features["personal_greeting"] = any(
                greeting in cleaned_text.lower() for greeting in greetings
            )

        if "personal_closing" in features:
            closings = ["love", "best regards", "sincerely", "yours truly", "take care"]
            features["personal_closing"] = any(
                closing in cleaned_text.lower() for closing in closings
            )

        if "personal_relationship" in features:
            relationships = ["mom", "dad", "brother", "sister", "friend"]
            features["personal_relationship"] = any(
                rel in cleaned_text.lower() for rel in relationships
            )

        if "personal_meeting" in features:
            meeting_words = ["weekend", "meeting", "see you", "get together"]
            features["personal_meeting"] = any(
                word in cleaned_text.lower() for word in meeting_words
            )

        if "personal_tone" in features:
            personal_words = ["look forward", "hope", "excited", "happy"]
            features["personal_tone"] = any(
                phrase in cleaned_text.lower() for phrase in personal_words
            )

        # Add TF-IDF features
        tfidf_feature_names = [
            f"tfidf_{name}" for name in tfidf.get_feature_names_out()
        ]
        for i, name in enumerate(tfidf_feature_names):
            if name in features:
                features[name] = (
                    tfidf_features[0, i] if tfidf_features.shape[1] > i else 0
                )

        # Create a DataFrame with the features in the correct order
        all_features = pd.DataFrame([features])

        # Make prediction
        prediction = model.predict(all_features)[0]
        proba = model.predict_proba(all_features)[0][
            1
        ]  # Probability of being phishing (class 1)
        confidence = float(proba * 100)

        return {
            "is_phishing": bool(prediction),
            "confidence": confidence,
            "features": features,
            "error": None,
        }

    except Exception as e:
        error_msg = str(e)
        print(f"Error in predict_email: {error_msg}")
        return {
            "is_phishing": False,
            "confidence": 0,
            "features": {},
            "error": f"Error analyzing email: {error_msg}",
        }


# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Phishing Email Detector</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        .container {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        textarea {
            width: 100%;
            height: 200px;
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-family: Arial, sans-serif;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            display: block;
            margin: 20px auto;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 8px;
        }
        .error {
            color: #ff4444;
        }
        .success {
            color: #4CAF50;
        }
        .warning {
            color: #ff9800;
        }
        .progress-container {
            width: 100%;
            height: 20px;
            background-color: #f3f3f3;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-bar {
            height: 100%;
            background-color: #4CAF50;
            width: 0%;
            transition: width 0.3s ease-in-out;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4CAF50;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-left: 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .feature-list {
            list-style-type: none;
            padding: 0;
        }
        .feature-list h4 {
            margin: 10px 0 5px;
            color: #333;
        }
        .feature-list li {
            margin: 5px 0;
            padding: 5px;
            background-color: #fff;
            border-radius: 3px;
            border-left: 3px solid #4CAF50;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Phishing Email Detector</h1>
        <p>Paste an email below to check if it's a phishing attempt.</p>
        
        <form id="emailForm">
            <textarea id="emailText" rows="10" placeholder="Paste email content here..." required></textarea>
            <button type="submit">Analyze Email</button>
        </form>

        <div id="result" style="display: none;">
            <h2>Analysis Results</h2>
            <div class="progress-container">
                <div class="progress-bar" id="progressBar"></div>
            </div>
            <div id="analysisStatus">Analyzing email...</div>
            <div id="confidence" class="result-item"></div>
            <div id="features" class="result-item"></div>
        </div>
    </div>

    <script>
        // Progress stages
        const stages = [
            { text: 'Analyzing email...', progress: 0 },
            { text: 'Cleaning text...', progress: 20 },
            { text: 'Processing text...', progress: 40 },
            { text: 'Extracting features...', progress: 60 },
            { text: 'Making prediction...', progress: 80 },
            { text: 'Generating results...', progress: 100 }
        ];
        
        // Function to update progress
        function updateProgress(stageIndex) {
            const progressBar = document.getElementById('progressBar');
            const analysisStatus = document.getElementById('analysisStatus');
            const stage = stages[stageIndex];
            
            progressBar.style.width = `${stage.progress}%`;
            analysisStatus.textContent = stage.text;
        }

        document.getElementById('emailForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const emailText = document.getElementById('emailText').value;
            const resultDiv = document.getElementById('result');
            const confidenceDiv = document.getElementById('confidence');
            const featuresDiv = document.getElementById('features');
            
            // Show result container
            resultDiv.style.display = 'block';
            
            // Reset previous results
            confidenceDiv.textContent = '';
            featuresDiv.innerHTML = '';
            
            // Show initial progress
            updateProgress(0);
            
            // Start the progress animation
            let currentStage = 0;
            const progressInterval = setInterval(() => {
                if (currentStage < stages.length - 1) {
                    currentStage++;
                    updateProgress(currentStage);
                }
            }, 1000);
            
            fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email: emailText })
            })
            .then(response => response.json())
            .then(data => {
                // Clear the progress interval
                clearInterval(progressInterval);
                
                // Show results
                const analysisStatus = document.getElementById('analysisStatus');
                analysisStatus.textContent = data.is_phishing ? "Phishing Detected!" : "Legitimate Email";
                analysisStatus.className = data.is_phishing ? "error" : "success";
                
                // Show confidence
                confidenceDiv.textContent = `Confidence: ${data.confidence.toFixed(1)}%`;
                
                // Show features
                if (data.features) {
                    const featureList = document.createElement('ul');
                    featureList.className = 'feature-list';
                    
                    // Group features by type
                    const featureGroups = {
                        'Text Features': [],
                        'TF-IDF Features': []
                    };
                    
                    Object.entries(data.features).forEach(([key, value]) => {
                        const featureItem = document.createElement('li');
                        if (key.startsWith('tfidf_')) {
                            featureItem.textContent = `${key}: ${value.toFixed(4)}`;
                            featureGroups['TF-IDF Features'].push(featureItem);
                        } else {
                            featureItem.textContent = `${key}: ${value}`;
                            featureGroups['Text Features'].push(featureItem);
                        }
                    });
                    
                    // Add grouped features
                    Object.entries(featureGroups).forEach(([group, items]) => {
                        if (items.length > 0) {
                            const groupTitle = document.createElement('h4');
                            groupTitle.textContent = group;
                            featureList.appendChild(groupTitle);
                            items.forEach(item => featureList.appendChild(item));
                        }
                    });
                    
                    featuresDiv.appendChild(featureList);
                }
                
                if (data.error) {
                    analysisStatus.textContent = `Error: ${data.error}`;
                    analysisStatus.className = "error";
                }
            })
            .catch(error => {
                console.error('Error:', error);
                clearInterval(progressInterval);
                const analysisStatus = document.getElementById('analysisStatus');
                analysisStatus.textContent = "Error analyzing email. Please try again.";
                analysisStatus.className = "error";
            });
        });
    </script>
</body>
</html>
"""


@app.route("/")
def home():
    return HTML_TEMPLATE


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        email_text = data.get("email", "")

        if not email_text or not email_text.strip():
            return (
                jsonify(
                    {
                        "is_phishing": False,
                        "confidence": 0,
                        "features": {},
                        "error": "Email text is required",
                    }
                ),
                400,
            )

        print(f"Analyzing email of length: {len(email_text)}")
        print("Starting prediction...")

        @with_timeout(30)
        def safe_predict(email_text):
            try:
                # Log each step of the prediction process
                print("Step 1: Cleaning text...")
                cleaned_text = clean_text(email_text)
                print(f"Cleaned text length: {len(cleaned_text)}")

                print("Step 2: Preprocessing text...")
                processed_text = preprocess_text(cleaned_text)
                print(f"Processed text length: {len(processed_text)}")

                print("Step 3: Analyzing links...")
                num_links, url_mismatch = analyze_links(email_text)
                print(f"Found {num_links} links with {url_mismatch} suspicious URLs")

                print("Step 4: Creating features...")
                # Create TF-IDF features
                tfidf_features = tfidf.transform([processed_text])

                # Log TF-IDF features
                print("\nTF-IDF Features:")
                print(f"Feature names: {tfidf.get_feature_names_out()}")
                if app.debug:
                    print(f"Feature values shape: {tfidf_features.shape}")

                # Get all feature names the model expects
                all_feature_names = (
                    model.feature_names_in_
                    if hasattr(model, "feature_names_in_")
                    else []
                )

                # Initialize all features to 0
                features = {name: 0 for name in all_feature_names}

                # Set basic features
                if "text_length" in features:
                    features["text_length"] = len(cleaned_text)
                if "num_uppercase" in features:
                    features["num_uppercase"] = sum(
                        1 for c in cleaned_text if c.isupper()
                    )
                if "phishing_keyword_count" in features:
                    features["phishing_keyword_count"] = sum(
                        1 for word in phishing_keywords if word in cleaned_text.lower()
                    )

                # Add TF-IDF features
                tfidf_feature_names = [
                    f"tfidf_{name}" for name in tfidf.get_feature_names_out()
                ]
                for i, name in enumerate(tfidf_feature_names):
                    if name in features:
                        features[name] = (
                            tfidf_features[0, i] if tfidf_features.shape[1] > i else 0
                        )

                print("Step 5: Creating DataFrame...")
                all_features = pd.DataFrame([features])

                print("Step 6: Making prediction...")
                prediction = model.predict(all_features)[0]
                proba = model.predict_proba(all_features)[0][1]
                confidence = float(proba * 100)

                print("Prediction complete")
                return {
                    "is_phishing": bool(prediction),
                    "confidence": confidence,
                    "features": features,
                    "error": None,
                }
            except Exception as e:
                print(f"Error in prediction step: {str(e)}")
                raise

        try:
            result = safe_predict(email_text)
        except TimeoutError:
            return (
                jsonify(
                    {
                        "is_phishing": False,
                        "confidence": 0,
                        "features": {},
                        "error": "Prediction took too long to complete",
                    }
                ),
                500,
            )
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise

        # Ensure all required fields are present
        if "is_phishing" not in result:
            result["is_phishing"] = False
        if "confidence" not in result:
            result["confidence"] = 0
        if "features" not in result:
            result["features"] = {}

        print(f"Final result: {result}")
        return jsonify(result)

    except Exception as e:
        import traceback

        error_msg = str(e)
        print(f"Error in analyze endpoint: {error_msg}")
        print(f"Full traceback: {traceback.format_exc()}")
        return (
            jsonify(
                {
                    "is_phishing": False,
                    "confidence": 0,
                    "features": {},
                    "error": f"Server error: {error_msg}",
                }
            ),
            500,
        )


if __name__ == "__main__":
    print("Starting Phishing Email Detector...")
    print("Access the web interface at: http://localhost:5000")
    app.run(debug=True, port=5000)
