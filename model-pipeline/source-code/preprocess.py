# This file handles loading and preparing (preprocessing) the dataset

from sklearn.datasets import load_iris  # Built-in dataset
from sklearn.model_selection import train_test_split  # Splitting data
from sklearn.preprocessing import StandardScaler  # Scaling features

def load_and_preprocess():
    # Load dataset
    data = load_iris()

    # X = input features (measurements), y = target labels (flower types)
    X = data.data
    y = data.target

    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create scaler (used to normalize data)
    scaler = StandardScaler()

    # Fit scaler on training data and transform it
    X_train = scaler.fit_transform(X_train)

    # Only transform test data (do not fit again)
    X_test = scaler.transform(X_test)

    # Return processed data and scaler
    return X_train, X_test, y_train, y_test, scaler
