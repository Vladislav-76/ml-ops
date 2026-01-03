# ============================= Treelite model =============================

FROM continuumio/miniconda3:24.11.1-0 AS convert-model

COPY .conda/sklearn.yml conda-pack.yml
RUN conda env create -f conda-pack.yml

COPY misc /misc
RUN conda run -n triton_scripts python3 /misc/train_convert_fil.py

# ============================== tritonserver ==============================

FROM nvcr.io/nvidia/tritonserver:25.10-py3 AS tritonserver

COPY models_fil /models
COPY --from=convert-model checkpoint.tl /models/titanic/1/checkpoint.tl

ENTRYPOINT ["sh", "-c", "tritonserver --model-repository=/models"]
