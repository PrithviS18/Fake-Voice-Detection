# Import TensorFlow to load the trained deep learning model
import tensorflow as tf

# Import preprocessing function to convert audio into MFCC features
from .preprocess import preprocess_audio

# Import path where trained model is stored
from .config import MODEL_PATH


# ---------------------------------------------------
# Load Trained Model (Executed Once at Startup)
# ---------------------------------------------------
# This loads the trained CNN model into memory.
# IMPORTANT:
# - This happens only once when the file is imported.
# - The model stays in memory for all future predictions.
# - This makes inference fast and production-ready.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


# ---------------------------------------------------
# Prediction Function
# ---------------------------------------------------
# Takes file path of audio
# Returns classification result as dictionary
def predict(file_path):

    # ------------------------------------------
    # Step 1: Preprocess Audio
    # ------------------------------------------
    # Convert raw .wav file into MFCC features
    # Shape after preprocessing:
    # (1, n_mfcc, time_frames, 1)
    features = preprocess_audio(file_path)


    # ------------------------------------------
    # Step 2: Run Model Inference
    # ------------------------------------------
    # model.predict returns:
    # [[probability]]
    #
    # Example:
    # [[0.87]]
    #
    # We extract the scalar value using [0][0]
    prediction = model.predict(features)[0][0]


    # ------------------------------------------
    # Step 3: Convert Probability to Label
    # ------------------------------------------
    # Since this is binary classification:
    #
    # Output range: 0 → 1
    # > 0.5  → Fake
    # <= 0.5 → Real

    if prediction > 0.5:
        return {
            "label": "Fake",
            "confidence": float(prediction)
        }
    else:
        return {
            "label": "Real",
            # If prediction is 0.2 → 80% real
            "confidence": float(1 - prediction)
        }