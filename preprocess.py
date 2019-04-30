import os
from argparse import ArgumentParser

from src.data import preprocess_data

argparser = ArgumentParser()
argparser.add_argument("--data_path", help="Path to data pickle file.")
argparser.add_argument("--data_name", choices=["seizure", "normal"], help="Name of data.")
argparser.add_argument("--output_path", help="Path to directory where the preprocessed data will be saved.")
argparser.add_argument("--feature_size", default=128, type=int, help="Batch size.")

args = argparser.parse_args()

data_path = args.data_path
assert data_path, "Data path is required"

data_name = args.data_name
assert data_name, "Data name is required"

output_path = args.output_path
assert output_path and os.path.isdir(output_path), "Output path is required"

preprocess_data(
    data_fname=data_path,
    data_name=data_name,
    output_dir=output_path,
    feature_size=args.feature_size
)
