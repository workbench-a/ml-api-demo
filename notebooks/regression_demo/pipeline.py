from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from regression_demo.processing import features
from regression_demo.processing import preprocessors as pp
from regression_demo.config import config

import logging

# ?? why is this even here?
_logger = logging.getLogger(__name__)

reg_demo_pipe = Pipeline(
    [
        (
            "demo_transformer",
            features.DemoTransformer(config.FEATURES),
        ),
        (
            "drop_features",
            pp.DropUnecessaryFeatures(config.DROP_FEATURES),
        ),
        (
            "linear model",
            LinearRegression(), # no random state to set here
        )
    ]
)