from simple_regression_demo.processing.local_data_management import load_dataset, save_pipeline
from simple_regression_demo import pipeline
from simple_regression_demo.processing.transform_target import half_scale
from simple_regression_demo.config import config
from simple_regression_demo import __version__ as _version

import logging

_logger = logging.getLogger(__name__)

def run_training() -> None:
    """Train the model."""
    
    # read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)
    X_train = data[config.FEATURES]
    y_train = data[config.TARGET]
    
    # transform amd fit
    y_train = half_scale(target=y_train)
    pipeline.reg_demo_pipe.fit(X_train[config.FEATURES], y_train)
    
    _logger.info(f"saving model version: {_version}")
    save_pipeline(pipeline_to_persist=pipeline.reg_demo_pipe)
    
if __name__=="__main__":
    run_training()