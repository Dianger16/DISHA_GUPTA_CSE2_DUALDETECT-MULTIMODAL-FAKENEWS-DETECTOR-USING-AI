import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

# Download NLTK resources if not already present
nltk.download('punkt')
nltk.download('stopwords')

# File paths
FAKE_FILE = 'gossipcop_fake.csv'
REAL_FILE = 'gossipcop_real.csv'

# Load datasets
print("Reading datasets...")
fake_df = pd.read_csv(FAKE_FILE)
real_df = pd.read_csv(REAL_FILE)

# Combine datasets
fake_df['label'] = 0  # Fake news label
real_df['label'] = 1  # Real news label

combined_df = pd.concat([fake_df, real_df], ignore_index=True)

print(f"Combined dataset shape: {combined_df.shape}")

# Check if 'title' exists
if 'title' not in combined_df.columns:
    raise KeyError("The dataset does not have a 'title' column. Please check column names.")

# Text cleaning function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()  # Split into words
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)

# Apply cleaning to the 'title' column
print("Preprocessing text data...")
combined_df['cleaned_title'] = combined_df['title'].apply(clean_text)

# Save the cleaned dataset to a new file
combined_df.to_csv('cleaned_combined_dataset.csv', index=False)

print("Text preprocessing completed and saved to 'cleaned_combined_dataset.csv'")
