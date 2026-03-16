# Import NumPy for numerical operations
import numpy as np

# Import evaluation metrics from sklearn
# classification_report -> precision, recall, F1-score
# roc_auc_score -> measures model discrimination ability
# confusion_matrix -> shows TP, TN, FP, FN counts
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


# ---------------------------------------------------
# Function to Evaluate Trained Model
# ---------------------------------------------------
# model   -> trained CNN model
# X_test  -> test feature data
# y_test  -> true labels (0 = Real, 1 = Fake)
def evaluate_model(model, X_test, y_test):

    # ------------------------------------------
    # Step 1: Get Predicted Probabilities
    # ------------------------------------------
    # model.predict returns probabilities between 0 and 1
    # Example output:
    # [[0.91],
    #  [0.12],
    #  [0.76], ...]
    y_pred_probs = model.predict(X_test)


    # ------------------------------------------
    # Step 2: Convert Probabilities to Class Labels
    # ------------------------------------------
    # If probability > 0.5 → Fake (1)
    # Else → Real (0)
    #
    # .astype(int) converts True/False into 1/0
    y_pred = (y_pred_probs > 0.5).astype(int)


    # ------------------------------------------
    # Step 3: Print Classification Report
    # ------------------------------------------
    # Shows:
    # - Precision
    # - Recall
    # - F1-score
    # - Support (number of samples per class)
    #
    # Important for deepfake detection:
    # - High recall for Fake → catches most deepfakes
    # - High precision → avoids false accusations
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


    # ------------------------------------------
    # Step 4: ROC-AUC Score
    # ------------------------------------------
    # ROC-AUC measures how well the model separates classes.
    #
    # Value ranges:
    # 0.5  → random guessing
    # 0.7+ → decent
    # 0.8+ → strong
    # 0.9+ → excellent
    #
    # Uses probabilities instead of thresholded labels.
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_probs))


    # ------------------------------------------
    # Step 5: Confusion Matrix
    # ------------------------------------------
    # Layout:
    #
    # [[TN  FP]
    #  [FN  TP]]
    #
    # TN → True Real detected correctly
    # TP → True Fake detected correctly
    # FP → Real predicted as Fake
    # FN → Fake predicted as Real (dangerous case)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))