import pandas as pd
import re, nltk
from nltk.corpus import stopwords
import email.utils
from urllib.parse import urlparse

# Load and clean the data
df = pd.read_csv("phishing-email-dataset/phishing_email.csv")

# Clean text: lowercase and strip whitespace
df["text_combined"] = df["text_combined"].str.lower().str.strip()

# Display data info
print("DataFrame Info:")
print(df.info())
print("\nLabel distribution:")
print(df["label"].value_counts())

# Remove stopwords from text_combined
stop = set(stopwords.words("english"))
df["text_cleaned"] = df["text_combined"].apply(
    lambda s: " ".join(w for w in re.findall(r"\w+", s.lower()) if w not in stop)
)

# Extract metadata features
print("\nExtracting metadata features...")

# 1. Sender domain analysis
def extract_sender_domain(email_str):
    try:
        parsed = email.utils.parseaddr(email_str)
        if parsed[1]:
            domain = parsed[1].split('@')[-1]
            return domain.lower()
    except:
        pass
    return ''

df['sender_domain'] = df['text_combined'].apply(extract_sender_domain)

# 2. Subject keyword flags
phishing_keywords = [
    'urgent', 'account', 'password', 'verify', 'security',
    'update', 'confirm', 'login', 'alert', 'warning'
]

def count_phishing_keywords(text):
    return sum(1 for word in phishing_keywords if word in text.lower())

df['phishing_keyword_count'] = df['text_combined'].apply(count_phishing_keywords)

# 3. Link analysis
def analyze_links(text):
    # Count total links
    links = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
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
df[['num_links', 'url_mismatch']] = df['text_combined'].apply(
    analyze_links
).apply(pd.Series)

# 4. Other heuristics
df['exclamation_count'] = df['text_combined'].str.count('!')
df['official_domain_mention'] = df['text_combined'].str.contains(r'@officialdomain', case=False).astype(int)

# Display new features
print("\nNew features added:")
print(df[['sender_domain', 'phishing_keyword_count', 'num_links', 'url_mismatch',
         'exclamation_count', 'official_domain_mention']].head())

# Display statistics of numeric features
print("\nStatistics of numeric features:")
print(df[['phishing_keyword_count', 'num_links', 'url_mismatch',
         'exclamation_count', 'official_domain_mention']].describe())
