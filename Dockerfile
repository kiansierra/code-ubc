FROM nvcr.io/nvidia/pytorch:23.10-py3 as base

COPY requirements_dev.txt .
RUN pip install -r requirements_dev.txt

COPY requirements.txt .
RUN pip install -r requirements.txt