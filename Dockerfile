#FROM ubuntu:22.04
FROM python:3.10-slim-bullseye
# COPY pyproject.toml /app/pyproject.toml
COPY pyproject.toml .
# COPY poetry.lock /app/poetry.lock
COPY poetry.lock .
# COPY ./daisybell /app/daisybell
COPY ./daisybell .
RUN pip install poetry==1.4.1
# ENV PATH="${PATH}:/root/.local/bin"
RUN apt-get update && apt-get upgrade -y 
# RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.1.15
RUN poetry config virtualenvs.create false
RUN poetry install 
WORKDIR ./daisybell
CMD ["python3", "daisybell/test.py"]