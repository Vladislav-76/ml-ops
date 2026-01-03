import sys
import os
from uuid import uuid4

import numpy as np
from tritonclient.http import InferenceServerClient, InferInput
from tritonclient.utils import InferenceServerException, np_to_triton_dtype

from utils import create_client

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
misc_dir = os.path.join(root_dir, "misc")
sys.path.append(misc_dir)

from train_convert_fil import X_test, y_test


if __name__ == "__main__":
    input_array = X_test[60:63].to_numpy(np.float32)
    output_array = y_test[60:63].to_numpy()
    print(input_array.shape)

    print("----- INPUT ------")
    print(input_array)
    input_0 = InferInput(
        "input__0",
        list(input_array.shape),
        np_to_triton_dtype(input_array.dtype)
    )
    input_0.set_data_from_numpy(input_array)
    inputs = [input_0]

    triton_client = create_client()

    try:
        resp = triton_client.infer("titanic", inputs, request_id=str(uuid4()))
    except InferenceServerException as ex:
        print(ex)
    else:
        print("----- OUTPUT -----")
        result = resp.as_numpy("output__0")
        match = result == output_array
        print(f"{result=}")
        print(f"{match=}")

    finally:
        triton_client.close()
