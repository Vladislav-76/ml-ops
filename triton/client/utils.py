from tritonclient.http import InferenceServerClient


def create_client(url: str = "localhost:8000") -> InferenceServerClient:
    triton_client = InferenceServerClient(url)
    return triton_client
