import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_classifier(X_train, y_train, model_type="rf", random_state=42):
    """
    Trains a classification model.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        model_type: "rf" for RandomForest, "lr" for LogisticRegression.
        random_state: Seed for reproducibility.
        
    Returns:
        Trained model instance.
    """
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model_type == "lr":
        model = LogisticRegression(random_state=random_state, max_iter=1000)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    """
    Saves a trained model to the specified filepath.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Loads a trained model from the specified filepath.
    """
    return joblib.load(filepath)
