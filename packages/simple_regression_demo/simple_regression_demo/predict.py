import numpy as np
import pandas as pd

from simple_regression_demo.processing.local_data_management import load_pipeline
from simple_regression_demo.processing.transform_target import double_scale
from simple_regression_demo.config import config
from simple_regression_demo import __version__ as _version

import logging
import typing as t

_logger = logging.getLogger(__name__)

# add type hints 
def make_prediction(*, input_data: t.Union[pd.DataFrame, dict],) -> dict:
    """Make prediction using a saved model pipeline."""
    # data = pd.DataFrame(input_data) # deprecated: already in data frame format
#     print(data)
#     validated_data = validate_inputs(input_data=data)
#     prediction = _price_pipe.predict(validated_data[config.FEATURES])

    pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
    _price_pipe = load_pipeline(file_name=pipeline_file_name)

    validated_data = input_data
    prediction = _price_pipe.predict(validated_data[config.FEATURES])
    output = double_scale(target=prediction)
    results = {"predictions": output, "version": _version}
    
    _logger.info(
        f"Making predictions with model version: {_version} "
        f"Inputs: {validated_data} "
        f"Predictions: {results}"
    )
    
    return results