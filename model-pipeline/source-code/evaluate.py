# This file evaluates how well the trained model performs

import pickle
from sklearn.metrics import accuracy_score  # Measures performance
from preprocess import load_and_preprocess

def evaluate_model():
    # Get test data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

    # Load saved model
    with open("../model/model.pkl", "rb") as f:
        model, scaler = pickle.load(f)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {acc:.2f}")

if __name__ == "__main__":
    evaluate_model()
