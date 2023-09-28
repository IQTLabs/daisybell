FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt update && apt upgrade -y
RUN apt install -y python3 python3-pip

WORKDIR /daisybell
COPY pyproject.toml .
COPY poetry.lock .
COPY README.rst .
COPY ./daisybell ./daisybell

RUN pip install poetry==1.5.1
RUN poetry config virtualenvs.create false
RUN poetry install

ENTRYPOINT ["daisybell"]
