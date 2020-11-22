import numpy as np
import pandas as pd
import typing as t

def half_scale(*, target):
    """Divide target elementwise by two."""
    return 0.5*target

def double_scale(*, target):
    """Divide target elementwise by two."""
    return 2*target