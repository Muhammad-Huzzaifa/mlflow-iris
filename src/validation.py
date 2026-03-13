import numpy as np

EXPECTED_FEATURES = 4

def validate_payload(data):

    data = np.array(data)

    if len(data.shape) != 2:
        raise ValueError("Input must be 2D")

    if data.shape[1] != EXPECTED_FEATURES:
        raise ValueError("Invalid number of features")

    if np.isnan(data).any():
        raise ValueError("Input contains missing values")

    return data
