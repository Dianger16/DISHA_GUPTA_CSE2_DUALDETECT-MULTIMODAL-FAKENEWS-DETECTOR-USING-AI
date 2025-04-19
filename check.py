from tensorflow.keras.models import load_model

# Load your image model
model_path = "C:/Users/digup/OneDrive/Desktop/dataset/models/image_model.h5"
image_model = load_model(model_path)

# Print model summary
image_model.summary()
