# Import NumPy for numerical operations such as padding and normalization
import numpy as np

# Import librosa for audio loading and feature extraction
import librosa

# Import configuration constants:
# SAMPLE_RATE -> target audio sampling rate
# SAMPLES_PER_TRACK -> fixed number of samples (duration control)
# N_MFCC -> number of MFCC coefficients to extract
from .config import SAMPLE_RATE, SAMPLES_PER_TRACK, N_MFCC


# Function to preprocess an audio file and convert it into MFCC features
def preprocess_audio(file_path):

    # ---------------------------------------------------
    # Step 1: Load Audio File
    # ---------------------------------------------------
    # librosa.load:
    # - Loads audio file
    # - Resamples it to SAMPLE_RATE
    # - Converts to mono automatically
    # Returns:
    #   audio -> NumPy array of waveform values
    #   sr -> sampling rate used
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)


    # ---------------------------------------------------
    # Step 2: Ensure Fixed Duration
    # ---------------------------------------------------
    # Deep learning models require consistent input size.
    # So we either:
    #   - Pad shorter audio
    #   - Trim longer audio

    if len(audio) < SAMPLES_PER_TRACK:

        # Calculate how many samples are missing
        pad_width = SAMPLES_PER_TRACK - len(audio)

        # Pad with zeros at the end
        audio = np.pad(audio, (0, pad_width))

    else:
        # Trim audio to fixed duration
        audio = audio[:SAMPLES_PER_TRACK]


    # ---------------------------------------------------
    # Step 3: Extract MFCC Features
    # ---------------------------------------------------
    # MFCC (Mel-Frequency Cepstral Coefficients)
    # Converts raw waveform into frequency-domain features
    # that better represent human auditory perception.

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # At this stage:
    # mfcc shape = (N_MFCC, time_frames)


    # ---------------------------------------------------
    # Step 4: Normalize Features
    # ---------------------------------------------------
    # Normalization improves training stability.
    # It centers data around 0 and scales variance to 1.
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)


    # ---------------------------------------------------
    # Step 5: Add Dimensions for CNN
    # ---------------------------------------------------
    # CNN expects 4D input:
    # (batch_size, height, width, channels)
    #
    # Current shape:
    # (N_MFCC, time_frames)
    #
    # After adding dimensions:
    # (1, N_MFCC, time_frames, 1)

    mfcc = mfcc[np.newaxis, ..., np.newaxis]


    # Return processed MFCC ready for prediction or training
    return mfcc