from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import cv2

from src.preprocessing import load_image, select_green_channel
from src.features import extract_features
from src.models import build_model, train_pipeline, evaluate_model, save_model, load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diabetic Retinopathy Grading — Classical CV & ML Pipeline"
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["train", "evaluate", "inference"],
        help="Operation mode: train a new model, evaluate on labeled data, or run inference on new images"
    )
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to dataset root directory (class subfolders)")
    parser.add_argument("--model_path", type=str, default="artifacts/model.joblib", help="Path to save (train) or load (evaluate/inference) the model")
    parser.add_argument("--model_name", type=str, default="random_forest", choices=["random_forest", "svm", "decision_tree"])
    parser.add_argument("--input", type=str, help="Path to a single image or a folder of images (inference mode only)")
    return parser.parse_args()


def build_dataframe(data_path: Path) -> pd.DataFrame:
    """Scans class subfolders and returns a DataFrame of (image_path, label) pairs."""
    dataset = []
    for class_folder in sorted(data_path.iterdir()):
        if not class_folder.is_dir():
            continue
        label = class_folder.name
        images = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            images.extend(class_folder.glob(ext))
        for img_path in images:
            dataset.append((str(img_path), label))
    return pd.DataFrame(dataset, columns=["image_path", "label"])


def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Runs preprocessing and feature extraction on every image in the DataFrame."""
    records = []
    for _, row in df.iterrows():
        image = load_image(row["image_path"])
        green = select_green_channel(image)
        features = extract_features(green)
        records.append({"features": features, "label": row["label"]})
    return pd.DataFrame(records)


def save_results(content: str, output_path: Path):
    """Writes a string result (evaluation report or prediction summary) to a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    args = parse_args()

    # ── TRAIN ──────────────────────────────────────────────────────────────────
    if args.mode == "train":
        data_path = Path(args.data_dir)
        print(f"[train] Scanning dataset at: {data_path}")

        # Build image-label DataFrame from class subfolders
        df = build_dataframe(data_path)
        print(f"[train] Found {len(df)} images across {df['label'].nunique()} classes")

        # Extract features for every image
        print("[train] Extracting features...")
        tabular_df = extract_all_features(df)

        # Train the scaler + classifier pipeline
        print(f"[train] Training {args.model_name}...")
        pipeline, x_test, y_test = train_pipeline(args.model_name, tabular_df)

        # Save the trained pipeline to disk
        save_model(pipeline, args.model_path)
        print(f"[train] Model saved to {args.model_path}")

        # Print a quick baseline evaluation on the held-out test split
        acc, report = evaluate_model(pipeline, x_test, y_test)
        print(f"\n[train] Held-out test accuracy: {acc:.4f}")
        print(report)

    # ── EVALUATE ───────────────────────────────────────────────────────────────
    elif args.mode == "evaluate":
        data_path = Path(args.data_dir)
        model_path = Path(args.model_path)

        print(f"[evaluate] Loading model from: {model_path}")
        pipeline = load_model(model_path)

        print(f"[evaluate] Scanning labeled dataset at: {data_path}")
        df = build_dataframe(data_path)
        print(f"[evaluate] Found {len(df)} images")

        # Extract features
        print("[evaluate] Extracting features...")
        tabular_df = extract_all_features(df)

        x = np.vstack(tabular_df["features"].values)
        y = tabular_df["label"].values

        # Evaluate and collect report
        acc, report = evaluate_model(pipeline, x, y)
        print(f"\n[evaluate] Accuracy: {acc:.4f}")
        print(report)

        # Save results to results/evaluation_report.txt
        report_content = f"Accuracy: {acc:.4f}\n\n{report}"
        save_results(report_content, Path("results/evaluation_report.txt"))

    # ── INFERENCE ──────────────────────────────────────────────────────────────
    elif args.mode == "inference":
        model_path = Path(args.model_path)
        input_path = Path(args.input)

        print(f"[inference] Loading model from: {model_path}")
        pipeline = load_model(model_path)

        # Collect images: support both a single file and a folder of images
        if input_path.is_file():
            image_paths = [input_path]
        elif input_path.is_dir():
            image_paths = []
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                image_paths.extend(input_path.glob(ext))
        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")

        print(f"[inference] Running inference on {len(image_paths)} image(s)...")

        # Preprocess, extract features, and predict
        predictions = []
        for img_path in image_paths:
            image = load_image(str(img_path))
            green = select_green_channel(image)
            features = np.array(extract_features(green)).reshape(1, -1)
            label = pipeline.predict(features)[0]
            predictions.append({"image_path": str(img_path), "predicted_label": label})
            print(f"  {img_path.name} → Class {label}")

        # Save results to results/predictions.csv
        results_path = Path("results/predictions.csv")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(predictions).to_csv(results_path, index=False)
        print(f"\n[inference] Predictions saved to {results_path}")
    
