# =============================== conda-pack ===============================

FROM continuumio/miniconda3:24.11.1-0 AS conda-pack

COPY .conda/conda-pack.yml conda-pack.yml
RUN conda env create -f conda-pack.yml \
    && conda install -c conda-forge conda-pack \
    && conda-pack -n conda_pre_post --output conda-pack.tar.gz

# ============================== ResNet model ==============================

FROM python:3.12.3 AS convert-model

RUN pip install optimum[onnxruntime]==2.0.0 accelerate==1.11.0
RUN optimum-cli export onnx -m microsoft/resnet-50 --task image-classification /converted_model

# ============================== tritonserver ==============================

FROM nvcr.io/nvidia/tritonserver:25.10-py3 AS tritonserver

COPY models_ensemble /models
COPY --from=conda-pack conda-pack.tar.gz /app/conda-pack.tar.gz
COPY --from=convert-model /converted_model/model.onnx /models/inference/1/model.onnx
COPY --from=convert-model /converted_model/config.json /models/postprocess/1/config.json
RUN chmod 100 /models/entrypoint.sh

ENTRYPOINT ["sh", "-c", "/models/entrypoint.sh"]
