# what does main do
# 1. it starts by doing the argmax function
# 2. then we will take the inputs and pass them through the pipeline
from pathlib import Path

import argparse
def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", required = True, choices = ["train", "evaluate", "inference"])
  parser.add_argument("--input", required = True, help = "path of input image or folder")
  parser.add_argument("--output", required = True, help = "path to store output")
  parser.add_argument("--model", help = "model type")
  parser.add_argument("--model_path", help = "path of model you want to use")
  
  args = parser.parser_args()
  return args


if __name__ == "__main__":
  args = parse()
  input_path = Path("args.input")
  output_path = Path("args.output")
  
