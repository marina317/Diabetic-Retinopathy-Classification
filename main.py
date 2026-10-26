# what does main do
# 1. it starts by doing the argmax function
# 2. then we will take the inputs and pass them through the pipeline
from pathlib import Path
import pandas as pd

import argparse
def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", required = True, choices = ["train", "evaluate", "inference"])
  parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to dataset directory")
  parser.add_argument("--model_path", type=str, default="artifacts/model.joblib", help="Path to save or load model")
  parser.add_argument("--model_name", type=str, default="random_forest", choices=["random_forest", "svm", "decision_tree"])
  parser.add_argument("--input", type=str, help="Path to a single image or image folder for prediction")
    
  return parser.parse_args()
  


if __name__ == "__main__":
  args = parse()
  if args.mode == "train":
    # start by reading the path of the dataset
    data_path = Path(args.data_dir)
    # create a dataframe with image_path and label
    dataset = []
    for item in sorted(data_dir.iterdir()):
      class_name = item
      extensions = ["*png", "*jpg"]
      images = []
      for ext in extentions:
        images.extend(class_name.glob(ext))
      for img in images:
        dataset.append((img, class_name))
    df = pd.DataFrame(dataset, columns = ["image_path", "label"])
    
    # feature extraction loop
    
    # create a dataframe with features and labels
    # prepare x & y
    # call model training from models.py
    
  elif args.mode == "evaluate":
    # create a path of the model path
    # create a path of the evaluation data path
    # use preprocessing and feature extraction scripts
    # use the model to make predictions
    # use evaluation function from models.py
    # where should I put the results ?
    
  elif args.mode == "inference":
    # start by reading the model path
    # create a path of the images or image I want to predict
    # call preprocessing and feature extraction scripts
    # use the model to make predictions
    # where should I store the results?
    
