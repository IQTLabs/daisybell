FROM python:3.10-slim-bullseye
WORKDIR /daisybell
COPY pyproject.toml .
COPY poetry.lock .
COPY README.rst .
COPY ./daisybell ./daisybell
RUN pip install poetry==1.4.1
RUN apt-get update && apt-get upgrade -y
RUN poetry config virtualenvs.create false
RUN poetry install
ENTRYPOINT ["daisybell"]
