# Import preprocessing function
from .preprocess import preprocess_audio

# TensorFlow
import tensorflow as tf


# ---------------------------------------------------
# Prediction Function
# ---------------------------------------------------
def predict(file_path, model):

    # ------------------------------------------
    # Step 1: Preprocess Audio
    # ------------------------------------------
    features = preprocess_audio(file_path)


    # ------------------------------------------
    # Step 2: Run Model Inference
    # ------------------------------------------
    infer = model.signatures["serve"]

    output = infer(tf.constant(features))


    # Extract probability
    prediction = list(output.values())[0].numpy()[0][0]


    # ------------------------------------------
    # Step 3: Convert Probability to Label
    # ------------------------------------------
    if prediction > 0.5:
        return {
            "label": "Fake",
            "confidence": float(prediction)
        }
    else:
        return {
            "label": "Real",
            "confidence": float(1 - prediction)
        }