from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image, UnidentifiedImageError
import random

# Initialize Flask app
app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/*": {"origins": "*"}})

# Model Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'text_model.h5')
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'image_model.h5')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl')

# Load models and vectorizer
def load_model_safe(path):
    try:
        if os.path.exists(path):
            print(f"✅ Loading model from {path}")
            return load_model(path)
        else:
            print(f"❌ Model not found: {path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    return None

def load_vectorizer_safe(path):
    try:
        if os.path.exists(path):
            print(f"✅ Loading vectorizer from {path}")
            return joblib.load(path)
        else:
            print(f"❌ Vectorizer not found: {path}")
    except Exception as e:
        print(f"❌ Error loading vectorizer: {e}")
    return None

text_model = load_model_safe(TEXT_MODEL_PATH)
image_model = load_model_safe(IMAGE_MODEL_PATH)
vectorizer = load_vectorizer_safe(VECTORIZER_PATH)

if not text_model or not image_model or not vectorizer:
    raise SystemExit("❌ One or more models are missing!")

# Sample comparison table data
comparison_stats = [
    {"Model": "Naive Bayes", "Accuracy": 0.78, "Precision": 0.76, "Recall": 0.77, "F1-Score": 0.765},
    {"Model": "SVM", "Accuracy": 0.82, "Precision": 0.80, "Recall": 0.81, "F1-Score": 0.805},
    {"Model": "LSTM", "Accuracy": 0.88, "Precision": 0.86, "Recall": 0.87, "F1-Score": 0.865},
    {"Model": "BERT", "Accuracy": 0.92, "Precision": 0.90, "Recall": 0.91, "F1-Score": 0.905},
    {"Model": "GPT-4 (Our Model)", "Accuracy": 0.95, "Precision": 0.94, "Recall": 0.94, "F1-Score": 0.94}
]

# Real news samples
real_news_examples = [
    "WHO approves new COVID-19 vaccine for global use.",
    "NASA's rover successfully lands on Mars.",
    "UN reaches historic agreement on carbon emissions.",
    "Record voter turnout in state elections reported by Election Commission.",
    "Global economy shows signs of recovery, says World Bank."
]

# Home route with optional stats
@app.route('/')
def home():
    return render_template("index.html", models=comparison_stats)

# Image Preprocessing
def preprocess_image(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image, dtype=np.float32)
        image_array = preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except UnidentifiedImageError:
        raise ValueError("Invalid image format")

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'text' not in request.form or 'image' not in request.files:
        return jsonify({'error': 'Missing text or image'}), 400

    text_input = request.form['text']
    image_file = request.files['image']

    # Process text
    try:
        text_array = vectorizer.transform([text_input]).toarray()
        text_prediction = text_model.predict(text_array)
    except Exception as e:
        return jsonify({'error': f'Text model error: {str(e)}'}), 500

    # Process image
    try:
        image_array = preprocess_image(image_file)
        image_prediction = image_model.predict(image_array)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Image model error: {str(e)}'}), 500

    # Final score and label
    final_score = (float(text_prediction[0][0]) + float(image_prediction[0][0])) / 2
    label = "Fake" if final_score > 0.5 else "Real"

    response = {
        "prediction": label,
        "score": round(final_score, 2),
        "comparison_table": comparison_stats,
        "message": "✅ Prediction successful!"
    }

    # Suggest real news if prediction is fake
    if label == "Fake":
        response["suggested_real_news"] = random.choice(real_news_examples)

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
