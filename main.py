# what does main do
# 1. it starts by doing the argmax function
# 2. then we will take the inputs and pass them through the pipeline
from pathlib import Path

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
    #do something
    x = 5
  elif args.mode == "evaluate":
    x = 6
  elif args.mode == "inference":
    x = 7
    
