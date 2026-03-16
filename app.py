# Import FastAPI framework
from fastapi import FastAPI

# Allow requests from frontend
from fastapi.middleware.cors import CORSMiddleware

# Pydantic model for request body
from pydantic import BaseModel

from src.config import MODEL_PATH

# File handling utilities
import os
import uuid

# Library to download file from URL
import requests

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
# Request Body Schema
# ---------------------------------------------------
class AudioRequest(BaseModel):
    file: str


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
async def detect(data: AudioRequest):

    audio_url = data.file

    # Validate file type
    if not audio_url.endswith(".wav"):
        return {"error": "Only .wav audio files are supported"}

    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)

    # Generate unique filename
    temp_path = f"uploads/{uuid.uuid4()}.wav"

    try:

        # Download file from URL
        response = requests.get(audio_url, timeout=15)

        if response.status_code != 200:
            return {"error": "Failed to download audio file"}

        # Save file locally
        with open(temp_path, "wb") as f:
            f.write(response.content)

        # Run prediction using preloaded model
        result = predict(temp_path, app.state.model)

    except Exception as e:
        return {"error": str(e)}

    finally:

        # Delete temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return result