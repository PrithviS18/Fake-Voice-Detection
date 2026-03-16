# Import FastAPI framework
from fastapi import FastAPI, File, UploadFile

# Allow requests from frontend
from fastapi.middleware.cors import CORSMiddleware

# File handling utilities
import shutil
import os
import uuid

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
    allow_origins=["*"],  # change to frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


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

        # Run model prediction
        result = predict(temp_path)

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Delete temp file after prediction
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return result