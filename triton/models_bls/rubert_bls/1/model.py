import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoConfig
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device
            ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.logger = pb_utils.Logger
        tokenizer_path = f"{args['model_repository']}/{args['model_version']}/tokenizer"
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(tokenizer_path)
        self.config = AutoConfig.from_pretrained(tokenizer_path)

        # Создаем метрики классификации
        self.metric_family = pb_utils.MetricFamily(
            name="prediction_class_labels",
            description="The labels of predictions",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self.metric_classes = {}
        for class_name in self.config.id2label.values():
             self.metric_classes[class_name] = self.metric_family.Metric(labels={"class": class_name})

        print('Initialized...')

    async def execute(self, requests):# -> List[pb_utils.InferenceResponse]:
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them.
        # Reusing the same pb_utils.InferenceResponse object for multiple
        # requests may result in segmentation faults. You should avoid storing
        # any of the input Tensors in the class attributes as they will be
        # overridden in subsequent inference requests. You can make a copy of
        # the underlying NumPy array and store it if it is required.
        for request in requests:
            # Perform inference on the request and append it to responses
            # list...
            input_message = pb_utils.get_input_tensor_by_name(request, "INPUT").as_numpy()
            message = [msg[0].decode("UTF-8") for msg in input_message]
            self.logger.log(f"ID: {request.request_id()} | INPUT: {message}")
            tokenized_message = self.tokenizer(message, return_tensors="np")

            inference_request = pb_utils.InferenceRequest(
                model_name='inference',
                requested_output_names=['logits'],
                inputs=[pb_utils.Tensor("input_ids", tokenized_message["input_ids"]),
                        pb_utils.Tensor("token_type_ids", tokenized_message["token_type_ids"]),
                        pb_utils.Tensor("attention_mask", tokenized_message["attention_mask"])],
                preferred_memory=pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU, 0),
            )
            inference_response = await inference_request.async_exec()

            # Check if the inference response has an error
            if inference_response.has_error():
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(inference_response.error().message())
                    )
                )
            else:
                # Extract the output tensors from the inference response.
                logits = pb_utils.get_output_tensor_by_name(inference_response, "logits").as_numpy()
                class_ids = np.argmax(logits, axis=1)

                response_class = []
                for cls_id in class_ids:
                    class_name = self.config.id2label[cls_id]
                    response_class.append(class_name.encode("UTF-8"))
                    self.metric_classes[class_name].increment(1)
                self.logger.log(f"ID: {request.request_id()} | OUTPUT: {response_class}")

                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[pb_utils.Tensor("OUTPUT", np.array(response_class, dtype=object))]
                    )
                )

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
