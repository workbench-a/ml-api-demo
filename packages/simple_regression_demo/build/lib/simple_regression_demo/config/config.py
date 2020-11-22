import pathlib
import simple_regression_demo
import pandas as pd

PACKAGE_ROOT = pathlib.Path(simple_regression_demo.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"
DATASET_TESTING_DIR = PACKAGE_ROOT / "datasets" / "testing"
LOGGING_DIR = PACKAGE_ROOT / "logs"

# logs
LOG_FILE = "logs.txt"
LOG_MODE = "console" # "file" or "console"

# data
DATA_FILE = "data.csv"
TRAINING_DATA_FILE = "train.csv"
DEVELOPMENT_DATA_FILE = "dev.csv"
TESTING_DATA_FILE = "test.csv"
TARGET = "price"


# variables
FEATURES = [
    "square_footage",
    "total_land_area"
]

DROP_FEATURES = "total_land_area"

PIPELINE_NAME = "simple_regression_demo"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"
