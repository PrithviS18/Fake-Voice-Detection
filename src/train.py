# Import OS module for handling file paths and directory traversal
import os

# Import NumPy for numerical operations and array handling
import numpy as np

# Import train_test_split for splitting dataset into train/validation/test sets
from sklearn.model_selection import train_test_split

# Import callbacks for early stopping and saving best model during training
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import custom preprocessing function to extract MFCC features
from .preprocess import preprocess_audio

# Import function that builds the CNN model architecture
from .model import build_model

# Import path where trained model will be saved
from .config import MODEL_PATH


# Path to dataset directory containing "real" and "fake" folders
DATASET_PATH = "dataset"


# Function to load dataset and convert audio files into MFCC feature arrays
def load_dataset():

    # List to store feature matrices
    X = []

    # List to store corresponding labels (0 = real, 1 = fake)
    y = []

    # Enumerate assigns:
    # 0 -> "real"
    # 1 -> "fake"
    for label, folder in enumerate(["real", "fake"]):

        # Construct full folder path
        folder_path = os.path.join(DATASET_PATH, folder)

        # Loop through each audio file inside the folder
        for file in os.listdir(folder_path):

            # Construct full file path
            file_path = os.path.join(folder_path, file)

            # Extract MFCC features using preprocessing function
            mfcc = preprocess_audio(file_path)

            # preprocess_audio returns shape: (1, n_mfcc, time_steps, 1)
            # Remove first dimension to store properly
            X.append(mfcc[0])

            # Append corresponding label
            y.append(label)

    # Convert lists into NumPy arrays
    return np.array(X), np.array(y)


# ===============================
# Load Dataset
# ===============================

print("Loading dataset...")

# X contains MFCC features
# y contains labels (0 or 1)
X, y = load_dataset()


# Add extra channel dimension for CNN compatibility
# Shape becomes: (samples, n_mfcc, time_steps, 1)
X = X[..., np.newaxis]


# ===============================
# Split Dataset
# ===============================

# First split:
# 80% training + validation
# 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,          # ensures class balance
    random_state=42      # ensures reproducibility
)


# Second split:
# From training set, take 20% for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)


# ===============================
# Build CNN Model
# ===============================

# Pass input shape (n_mfcc, time_steps, channels)
model = build_model(X_train.shape[1:])


# ===============================
# Setup Callbacks
# ===============================

callbacks = [

    # Stop training if validation loss does not improve for 5 epochs
    EarlyStopping(
        patience=5,
        restore_best_weights=True
    ),

    # Save the best model based on validation performance
    ModelCheckpoint(
        MODEL_PATH,
        save_best_only=True
    )
]


# ===============================
# Train Model
# ===============================

model.fit(
    X_train,
    y_train,
    epochs=30,                 # Maximum training epochs
    batch_size=32,             # Number of samples per gradient update
    validation_data=(X_val, y_val),
    callbacks=callbacks        # Apply early stopping & checkpoint
)


# ===============================
# Save Final Model
# ===============================

# Save trained model to disk
model.save(MODEL_PATH, compile=False)

print("Model saved at", MODEL_PATH)