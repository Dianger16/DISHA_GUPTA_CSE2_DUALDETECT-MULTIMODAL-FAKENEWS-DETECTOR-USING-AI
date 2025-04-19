import pandas as pd

# Load cleaned dataset
df = pd.read_csv("cleaned_combined_dataset.csv")

# Count number of fake and real news samples
print("Class distribution:")
print(df['label'].value_counts())
