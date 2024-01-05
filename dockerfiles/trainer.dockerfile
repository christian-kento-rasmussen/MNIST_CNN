FROM python:3.11-slim  
# for x86 architecture --platform=linux/amd64

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY MNIST_CNN MNIST_CNN
COPY models/ models/
COPY reports/ reports/
COPY data/ data/

WORKDIR /
RUN pip install -e . --no-cache-dir

ENTRYPOINT ["python", "-u", "MNIST_CNN/train_model.py"]