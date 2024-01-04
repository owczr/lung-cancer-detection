FROM continuumio/miniconda3

RUN conda update -n base -c defaults conda

COPY conda_dependencies.yaml .
RUN conda env create -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run pip cache purge && \
    conda clean -a -y
