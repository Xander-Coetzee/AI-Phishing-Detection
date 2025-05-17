import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("phishing-email-dataset/phishing_email.csv")

# Function to find a good sample email
def get_sample_email(label_type, min_length=200, max_length=2000):
    # Filter by label (0 for legitimate, 1 for phishing)
    filtered_emails = df[df['label'] == label_type].copy()
    
    # Calculate text length
    filtered_emails['text_length'] = filtered_emails['text_combined'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)
    
    # Filter by length to get reasonably sized emails
    good_length_emails = filtered_emails[(filtered_emails['text_length'] >= min_length) & 
                                        (filtered_emails['text_length'] <= max_length)]
    
    if good_length_emails.empty:
        # If no emails in the desired range, just get the longest ones
        good_length_emails = filtered_emails.sort_values('text_length', ascending=False).head(10)
    
    # Randomly select one email from the filtered set
    if not good_length_emails.empty:
        sample_email = good_length_emails.sample(1).iloc[0]
        return sample_email
    else:
        return None

# Get sample emails
legitimate_sample = get_sample_email(label_type=0)
phishing_sample = get_sample_email(label_type=1)

# Save legitimate email
if legitimate_sample is not None:
    print("\n===== LEGITIMATE EMAIL SAMPLE =====")
    print(f"Email ID: {legitimate_sample.name}")
    print(f"Label: {legitimate_sample['label']} (0 = Legitimate)")
    print(f"Length: {legitimate_sample['text_length']} characters")
    
    # Save to a file
    with open('legitimate_email_sample.txt', 'w', encoding='utf-8') as f:
        f.write(f"Email ID: {legitimate_sample.name}\n")
        f.write(f"Label: {legitimate_sample['label']} (0 = Legitimate)\n\n")
        f.write("EMAIL CONTENT:\n")
        f.write("=" * 80 + "\n")
        f.write(str(legitimate_sample['text_combined']))
        f.write("\n" + "=" * 80)
    
    print("\nSample legitimate email saved to 'legitimate_email_sample.txt'")
else:
    print("No suitable legitimate emails found in the dataset.")

# Save phishing email
if phishing_sample is not None:
    print("\n===== PHISHING EMAIL SAMPLE =====")
    print(f"Email ID: {phishing_sample.name}")
    print(f"Label: {phishing_sample['label']} (1 = Phishing)")
    print(f"Length: {phishing_sample['text_length']} characters")
    
    # Save to a file
    with open('phishing_email_sample.txt', 'w', encoding='utf-8') as f:
        f.write(f"Email ID: {phishing_sample.name}\n")
        f.write(f"Label: {phishing_sample['label']} (1 = Phishing)\n\n")
        f.write("EMAIL CONTENT:\n")
        f.write("=" * 80 + "\n")
        f.write(str(phishing_sample['text_combined']))
        f.write("\n" + "=" * 80)
    
    print("\nSample phishing email saved to 'phishing_email_sample.txt'")
else:
    print("No suitable phishing emails found in the dataset.")
