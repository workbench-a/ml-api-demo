import pandas as pd
import numpy as np
import os
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from regression_demo.config import config
from regression_demo import __version__ as _version

import logging
import typing as t

_logger = logging.getLogger(__name__)

def rel_path(rel_path_arr):
    """Return an os dependent relative path"""
    return os.path.join(os.path.curdir, *rel_path_arr)

def create_regression_demo_data(*, mode="production", n_samples=100, file_name=config.DATA_FILE) -> None:
    """
    Create data for regression demo. Supported modes are 'testing' 
    (for testing purposes) and 'production' (for all other data).
    """
    X = (np.array([[j + 10*(i+1)*np.random.randn() 
                    for i in range(0, 2)] for j in range(0, n_samples)]))
    y = np.array([3*i + np.random.randn() + 30 for i in range(0, n_samples)])
    data_dict = ({config.FEATURES[0]: X[:,0], config.FEATURES[1]: X[:,1], 
                  config.TARGET: y})
    data = pd.DataFrame(data=data_dict)
    if(mode == "production"):
        dataset_dir = config.DATASET_DIR
    elif(mode == "testing"):
        dataset_dir = config.DATASET_TESTING_DIR
    else:
        # throw exception: mode not supported
        pass
    data.to_csv(f"{dataset_dir}/{file_name}", index=False)

def load_dataset(*, mode="production", file_name: str) -> pd.DataFrame:
    """
    Load the specified dataset. Supported modes are 'testing' 
    (for testing purposes) and 'production' (for all other data).
    """
    if(mode == "production"):
        _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    elif(mode == "testing"):
        _data = pd.read_csv(f"{config.DATASET_TESTING_DIR}/{file_name}")
    else:
        # throw exception: mode not supported
        pass
    return _data

def make_splits(*, file_name=config.DATA_FILE, mode="production", split_type="train_test", test_size=0.3, random_state=3) -> None:
    """
    Make train and dev set splits. Supported modes are 'testing' (for testing purposes) 
    and 'production' (for all other data). Supported split types are 'train_dev' and, 
    the default, 'train_test.'
    """
    data = load_dataset(mode=mode, file_name=file_name)
    X_train, X_test, y_train, y_test = train_test_split(data[config.FEATURES],
                                                        data[config.TARGET],
                                                        test_size=test_size,
                                                        # set the seed here:
                                                        random_state=random_state)
    first_dataset_type = config.TRAINING_DATA_FILE
    if(split_type == "train_test"):
        second_dataset_type = config.TESTING_DATA_FILE
    elif(split_type == "train_dev"):
        second_dataset_type = config.DEVELOPMENT_DATA_FILE
    else:
        # throw exception: split type not supported
        pass
    if(mode == "production"):
        dataset_dir = config.DATASET_DIR
    elif(mode == "testing"):
        dataset_dir = config.DATASET_TESTING_DIR
    else:
        # throw exception: mode not supported
        pass      
    train_data = X_train.join(y_train)
    test_data = X_test.join(y_test)
    train_data.to_csv(f"{dataset_dir}/{first_dataset_type}", index=False)
    test_data.to_csv(f"{dataset_dir}/{second_dataset_type}", index=False)

def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipline."""
    # Prepare versioned save file name
    save_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    # ?? not immediately clear where immediate previous version is 
    # preserved 
    # for differential testing
    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    _logger.info(f"saved pipeline: {save_file_name}")

def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""
    file_path = config.TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model

def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    
    This is to ensure that there is a one-to-oneds 
    mapping between the package version and the model
    version to be imported and used by other applications.
    However, we do also include the immediate previous
    pipelines version for differential testing purposes.
    """
    do_not_delete = files_to_keep + ['__init__.py']
    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            print(config.TRAINED_MODEL_DIR)
            print(model_file)
            model_file.unlink()