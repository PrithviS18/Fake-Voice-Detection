# Import FastAPI framework
from fastapi import FastAPI, File, UploadFile

# Allow requests from frontend
from fastapi.middleware.cors import CORSMiddleware

from src.config import MODEL_PATH

# File handling utilities
import shutil
import os
import uuid

# TensorFlow for model loading
import tensorflow as tf

# Import prediction function
from src.predict import predict


# ---------------------------------------------------
# Create FastAPI application
# ---------------------------------------------------
app = FastAPI()


# ---------------------------------------------------
# Enable CORS
# ---------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------
# Load model at server startup
# ---------------------------------------------------
@app.on_event("startup")
def load_model():

    print("Loading model from:", MODEL_PATH)

    # Load TensorFlow SavedModel
    app.state.model = tf.saved_model.load(MODEL_PATH)

    print("Model loaded successfully")


# ---------------------------------------------------
# POST Endpoint: /detect
# ---------------------------------------------------
@app.post("/detect")
async def detect(audio: UploadFile = File(...)):

    # Validate file type
    if not audio.filename.endswith(".wav"):
        return {"error": "Only .wav audio files are supported"}

    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)

    # Generate unique filename
    temp_path = f"uploads/{uuid.uuid4()}.wav"

    try:

        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        # Run prediction using preloaded model
        result = predict(temp_path, app.state.model)

    except Exception as e:
        return {"error": str(e)}

    finally:

        # Delete temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return result