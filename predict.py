import joblib

# Load the trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_news(news_title):
    """Predict if a news title is fake or real."""
    news_title = [news_title]  # Convert to list
    news_vec = vectorizer.transform(news_title)  # Transform using TF-IDF
    prediction = model.predict(news_vec)[0]  # Get prediction (0 = Fake, 1 = Real)
    return "Real News" if prediction == 1 else "Fake News"

# Example Usage
news_title = input("Enter a news title: ")
print("Prediction:", predict_news(news_title))

