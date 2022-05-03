ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.03-py3
FROM ${FROM_IMAGE_NAME}

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN pip install --global-option="--cpp_ext" --global-option="--cuda_ext" git+git://github.com/NVIDIA/apex.git#egg=apex

WORKDIR /workspace/hourglass/

ENV NEPTUNE_API_TOKEN eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMTcxMzI2My1jOTY1LTQ5MjAtOGMzNC1jNmNhMzRlOGI3MGUifQ==
ENV NEPTUNE_PROJECT syzymon/trax
ENV NEPTUNE_TOKEN eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMTcxMzI2My1jOTY1LTQ5MjAtOGMzNC1jNmNhMzRlOGI3MGUifQ==
RUN pip install neptune-client
RUN pip install tokenizers 
