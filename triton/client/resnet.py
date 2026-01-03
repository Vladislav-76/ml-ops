from uuid import uuid4

import numpy as np
from datasets import load_dataset
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers import AutoImageProcessor
from tritonclient.http import InferenceServerClient, InferInput
from tritonclient.utils import InferenceServerException, np_to_triton_dtype

from utils import create_client


if __name__ == "__main__":

    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    inputs = processor(image, return_tensors="np")

    print("----- INPUT ------")
    print(f"pixel_values: {inputs["pixel_values"].shape}")

    # У нашей модели "resnet" есть один вход - INPUT
    input_0 = InferInput(
        "INPUT",
        list(inputs["pixel_values"].shape),
        np_to_triton_dtype(inputs["pixel_values"].dtype)
    )
    input_0.set_data_from_numpy(inputs["pixel_values"])

    triton_client = create_client()

    try:
        resp = triton_client.infer("resnet", [input_0], request_id=str(uuid4()))
    except InferenceServerException as ex:
        print(ex)
    else:
        print("----- OUTPUT -----")
        response_class = resp.as_numpy("OUTPUT").astype("U13")
        print(response_class)
    finally:
        triton_client.close()
