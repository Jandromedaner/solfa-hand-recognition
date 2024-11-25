import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_model(data_path="data/solfa_data.pkl", model_path="data/solfa_model.pkl"):
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    X = np.array(data['features'])
    y = np.array(data['labels'])

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Print accuracy
    print(f"Model accuracy: {model.score(X_test, y_test):.2f}")
    return model
