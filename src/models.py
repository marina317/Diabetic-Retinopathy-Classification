from pathlib import Path
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


def build_model(model_name="random_forest"):
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, max_depth=20, class_weight='balanced', random_state=42
        )
    elif model_name == "svm":
        return SVC(
            kernel="rbf", class_weight='balanced', random_state=42
        )
    elif model_name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=20, class_weight='balanced', random_state=42
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train_pipeline(model_name, df):
    x = np.vstack(df["features"].values)
    y = df["label"].values
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    

    classifier = build_model(model_name)
    
   
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])
    
    # Train pipeline on train data
    pipeline.fit(x_train, y_train)
    
    return pipeline, x_test, y_test


def evaluate_model(model, x_val, y_val):
    predictions = model.predict(x_val)
    acc = accuracy_score(y_val, predictions)
    report = classification_report(y_val, predictions)
    return acc, report


def save_model(model, file_path):
    save_path = Path(file_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Saved successfully to {save_path}")


def load_model(file_path):
    load_path = Path(file_path)
    if not load_path.exists():
        raise FileNotFoundError(f"File not found at {load_path}")
    model = joblib.load(load_path)
    print(f"Loaded successfully from {load_path}")
    return model
