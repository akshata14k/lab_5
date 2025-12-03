import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from google.cloud import storage
import joblib
from datetime import datetime
import os


def load_data():
    """Load the Breast Cancer dataset and return features and target."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y


def build_model():
    """
    Build a simple ML pipeline:
    - Standardize features
    - Train logistic regression classifier
    """
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    return pipe


def train_model(X, y, test_size=0.2, random_state=42):
    """Split data, train model, and return fitted model + accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.4f}")

    return model, acc


def save_model_to_gcs(model, bucket_name, blob_name):
    """Save the trained model object to a GCS bucket as a joblib file."""
    local_path = "model.joblib"
    joblib.dump(model, local_path)
    print(f"Saved model locally to {local_path}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)

    print(f"Uploaded model to gs://{bucket_name}/{blob_name}")


def main():
    X, y = load_data()
    model, acc = train_model(X, y)

    bucket_name = os.environ.get("GCS_MODEL_BUCKET")
    if not bucket_name:
        raise ValueError(
            "Environment variable GCS_MODEL_BUCKET is not set. "
            "Set it in your GitHub Actions workflow or local env."
        )

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    blob_name = f"trained_models/breast_cancer_model_{timestamp}.joblib"

    save_model_to_gcs(model, bucket_name, blob_name)


if __name__ == "__main__":
    main()
