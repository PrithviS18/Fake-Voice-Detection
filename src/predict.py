# Import preprocessing function to convert audio into MFCC features
from .preprocess import preprocess_audio


# ---------------------------------------------------
# Prediction Function
# ---------------------------------------------------
# Takes file path of audio
# Takes preloaded model from FastAPI
# Returns classification result
def predict(file_path, model):

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
    prediction = model.predict(features)[0][0]


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