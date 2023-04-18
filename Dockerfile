FROM ubuntu:22.04
COPY pyproject.toml /app/pyproject.toml
COPY poetry.lock /app/poetry.lock
COPY ./daisybell /app/daisybell
RUN apt-get update && apt-get upgrade 
RUN curl -sSL https://install.python-poetry.org | python3 - 
RUN poetry config virtualenvs.create false
WORKDIR /daisybell
CMD ["daisybell", "server?"]