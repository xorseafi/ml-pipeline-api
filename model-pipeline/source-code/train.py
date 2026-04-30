# This file trains the machine learning model and saves it

import pickle  # Used for saving the model
from sklearn.ensemble import RandomForestClassifier  # ML algorithm
from preprocess import load_and_preprocess  # Import preprocessing function

def train_model():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

    # Create the model
    model = RandomForestClassifier()

    # Train the model using training data
    model.fit(X_train, y_train)

    # Save the model and scaler to a file
    with open("../model/model.pkl", "wb") as f:
        pickle.dump((model, scaler), f)

    print("Model trained and saved successfully!")

# Run training when this file is executed
if __name__ == "__main__":
    train_model()
