# =============================== conda-pack ===============================

FROM continuumio/miniconda3:24.11.1-0 AS conda-pack

COPY .conda/conda-pack.yml conda-pack.yml
RUN conda env create -f conda-pack.yml \
    && conda install -c conda-forge conda-pack \
    && conda-pack -n conda_pre_post --output conda-pack.tar.gz

# =============================== BERT model ===============================

FROM python:3.12.3 AS convert-model

RUN pip install optimum[onnxruntime]==2.0.0 accelerate==1.11.0
RUN optimum-cli export onnx -m seara/rubert-tiny2-ru-go-emotions /converted_model

# ============================== tritonserver ==============================

FROM nvcr.io/nvidia/tritonserver:25.10-py3 AS tritonserver

COPY models_bls /models
COPY --from=conda-pack conda-pack.tar.gz /app/conda-pack.tar.gz
COPY --from=convert-model /converted_model/model.onnx /models/inference/1/model.onnx
COPY --from=convert-model /converted_model/*.json /models/rubert_bls/1/tokenizer/
COPY --from=convert-model /converted_model/*.txt /models/rubert_bls/1/tokenizer/

ENTRYPOINT ["sh", "-c", "tritonserver --model-repository=/models ${LAUNCH_PARAMETERS}"]
