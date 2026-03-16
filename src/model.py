# Import Sequential API to build a stack of neural network layers
from tensorflow.keras.models import Sequential

# Import required CNN layers
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)


# ---------------------------------------------------
# Function to Build CNN Model
# ---------------------------------------------------
# input_shape example:
# (40, 94, 1)
# where:
# 40  -> number of MFCC coefficients
# 94  -> time frames
# 1   -> single channel (grayscale-like input)
def build_model(input_shape):

    # Create a sequential model (layers stacked linearly)
    model = Sequential()

    # ---------------------------------------------------
    # Block 1
    # ---------------------------------------------------
    # Conv2D:
    # - 32 filters
    # - 3x3 kernel
    # - ReLU activation introduces non-linearity
    # Learns low-level audio patterns (basic frequency textures)
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation='relu',
            input_shape=input_shape
        )
    )

    # BatchNormalization:
    # - Stabilizes training
    # - Reduces internal covariate shift
    # - Speeds up convergence
    model.add(BatchNormalization())

    # MaxPooling:
    # - Reduces spatial dimensions
    # - Keeps strongest features
    # - Helps reduce overfitting
    model.add(MaxPooling2D((2, 2)))


    # ---------------------------------------------------
    # Block 2
    # ---------------------------------------------------
    # More filters (64) = deeper feature extraction
    # Learns more complex frequency relationships
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))


    # ---------------------------------------------------
    # Block 3
    # ---------------------------------------------------
    # 128 filters:
    # Detects high-level artifacts
    # Useful for detecting subtle synthetic voice patterns
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))


    # ---------------------------------------------------
    # Flatten Layer
    # ---------------------------------------------------
    # Converts 2D feature maps into 1D vector
    # Required before passing into Dense layers
    model.add(Flatten())


    # ---------------------------------------------------
    # Fully Connected Layer
    # ---------------------------------------------------
    # 256 neurons
    # Learns non-linear combinations of extracted features
    model.add(Dense(256, activation='relu'))

    # Dropout:
    # - Randomly drops 50% neurons during training
    # - Prevents overfitting
    model.add(Dropout(0.5))


    # ---------------------------------------------------
    # Output Layer
    # ---------------------------------------------------
    # 1 neuron
    # Sigmoid activation:
    # - Outputs probability between 0 and 1
    # - Suitable for binary classification
    model.add(Dense(1, activation='sigmoid'))


    # ---------------------------------------------------
    # Compile Model
    # ---------------------------------------------------
    # Adam optimizer:
    # - Adaptive learning rate
    # - Works well for most deep learning tasks
    #
    # Binary crossentropy:
    # - Standard loss function for binary classification
    #
    # Accuracy metric:
    # - Measures classification correctness
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )


    # Return compiled model
    return model