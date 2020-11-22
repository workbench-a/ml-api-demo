import pandas as pd

def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model for unprocessable values."""
    
    valudated_data = input_data.copy()
    
    # check for numerical variables with NA not seen during training
    
    return validated_data