FROM --platform=$TARGETPLATFORM python:3.12-slim

ARG PIP_NO_CACHE_DIR=1
ENV HVLOG_DIR=/data \
    AWS_REGION=us-east-1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# HERG submodule + agent code
COPY herg            ./herg
COPY agent           ./agent

RUN useradd -m runner
USER runner
VOLUME ["/data"]

EXPOSE 9000 8000      # 8000 for router if same image
CMD ["python", "-m", "agent.node"]
