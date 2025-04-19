import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

# Load dataset (example)
categories = ['sci.med', 'sci.space']
data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Create vectorizer & transform text
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model training complete! Saved as model.pkl and vectorizer.pkl.")
